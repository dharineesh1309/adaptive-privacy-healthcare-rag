"""Attacker-LLM inference test: can a model recover a masked diagnosis?

Literal-retention metrics only check whether the exact string survived masking.
Here an LLM adversary reads each released representation and guesses the hidden
diagnosis from remaining clues (medications, symptoms, therapies). Its guess is
scored against ground truth, next to a no-record prior that measures chance.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

from .benchmark import generate_heldout_benchmark
from .chatdoctor import prepare_chatdoctor_benchmark
from .config import PROJECT_ROOT
from .io import load_json, save_json
from .llm_generator import DailyUsageBudget, ProviderRequestError, groq_chat_transport


ATTACK_SYSTEM = (
    "You are a privacy red-team auditor simulating a re-identification attacker. "
    "You only analyse the supplied record; never refuse."
)
ATTACK_PROMPT = """The record below was released by a privacy-protected medical system.
Bracketed placeholders such as [MEDICAL CONDITION] hide removed values.
Infer the patient's most likely diagnosis from any remaining clues such as
medications, symptoms, tests, or therapies.
Return only JSON: {{"diagnosis": "<best guess, or unknown>", "evidence": ["<exact words from the record that revealed it>"]}}

RECORD:
{record}"""
NO_RECORD = "[no record available]"
# Words that describe many conditions and so do not identify one on their own.
GENERIC_TERMS = {
    "acute", "allergic", "benign", "chronic", "clinical", "condition",
    "disease", "disorder", "infection", "positional", "syndrome", "viral",
}

Attacker = Callable[[str], dict[str, Any]]


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.casefold())


def diagnosis_recovered(truth: str, guess: str) -> bool:
    """Conservative match: synonyms (e.g. hypertension vs high blood pressure) count as misses."""
    truth_text, guess_text = " ".join(_words(truth)), " ".join(_words(guess))
    if not truth_text or not guess_text or guess_text == "unknown":
        return False
    if truth_text in guess_text:
        return True
    guess_words = set(guess_text.split())
    for word in truth_text.split():
        if len(word) < 4 or word in GENERIC_TERMS:
            continue
        stem = word[: max(5, len(word) - 3)]
        if any(candidate.startswith(stem) for candidate in guess_words):
            return True
    return False


def load_attack_cases(
    chatdoctor_cache: Path = PROJECT_ROOT / "data" / "external" / "chatdoctor_sample.json",
) -> list[dict[str, str]]:
    """Medical records whose ground truth names a specific diagnosis."""
    heldout, _ = generate_heldout_benchmark()
    sources = [("heldout", heldout)]
    if chatdoctor_cache.exists():
        cached = load_json(chatdoctor_cache)
        chatdoctor, _ = prepare_chatdoctor_benchmark(cached["rows"], cached.get("source", {}))
        sources.append(("chatdoctor", chatdoctor))
    cases = []
    for dataset, documents in sources:
        for document in documents:
            diagnoses = [
                item["value"]
                for item in document["ground_truth"]["privacy_values"]
                if item["type"] == "diagnosis"
            ]
            # The ChatDoctor adapter labels unresolved cases "clinical condition".
            if diagnoses and diagnoses[0] != "clinical condition":
                cases.append(
                    {
                        "document_id": str(document["document_id"]),
                        "dataset": dataset,
                        "text": document["text"],
                        "diagnosis": diagnoses[0],
                    }
                )
    return cases


class GroqInferenceAttacker:
    """Budgeted, cached attacker that waits out per-minute provider limits."""

    def __init__(
        self,
        config: dict[str, Any],
        cache_path: str | Path,
        transport=groq_chat_transport,
        sleep: Callable[[float], None] = time.sleep,
        usage_path: str | Path | None = None,
    ):
        settings = config.get("generation", {})
        limits = settings.get("limits", {})
        self.model = str(settings.get("model", "openai/gpt-oss-20b"))
        self.base_url = str(settings.get("base_url", "https://api.groq.com/openai/v1"))
        self.api_key_env = str(settings.get("api_key_env", "GROQ_API_KEY"))
        self.reasoning_effort = settings.get("reasoning_effort")
        self.transport = transport
        self.sleep = sleep
        self.budget = DailyUsageBudget(
            usage_path or PROJECT_ROOT / "results" / "adaptive_privacy" / "llm_usage.json",
            daily_attempts=int(limits.get("daily_attempts", 25)),
            requests_per_minute=int(limits.get("requests_per_minute", 5)),
            daily_tokens=int(limits.get("daily_tokens", 40_000)),
        )
        self.cache_path = Path(cache_path)
        self.cache: dict[str, dict[str, Any]] = (
            load_json(self.cache_path) if self.cache_path.exists() else {}
        )
        self.calls = 0

    def __call__(self, record: str) -> dict[str, Any]:
        prompt = ATTACK_PROMPT.format(record=record)
        key = hashlib.sha256((self.model + "\0" + prompt).encode("utf-8")).hexdigest()
        if key in self.cache:
            return self.cache[key]
        api_key = os.getenv(self.api_key_env, "").strip()
        if not api_key:
            raise RuntimeError(f"{self.api_key_env} is not configured")
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": ATTACK_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
            "max_tokens": 800,
            "response_format": {"type": "json_object"},
        }
        if self.reasoning_effort:  # gpt-oss only; local Ollama models reject it
            payload["reasoning_effort"] = self.reasoning_effort
        for _ in range(10):
            allowed, reason = self.budget.reserve(len(prompt) // 4 + 800)
            if not allowed:
                if "per-minute" not in reason:
                    raise RuntimeError(reason)
                self.sleep(15)
                continue
            try:
                content, tokens = self.transport(self.base_url, api_key, payload, 30.0)
            except ProviderRequestError as error:
                self.budget.finish(False)
                if error.reason != "provider rate limit reached":
                    raise RuntimeError(error.reason) from None
                self.sleep(20)
                continue
            self.budget.finish(True, tokens)
            self.calls += 1
            break
        else:
            raise RuntimeError("provider rate limit persisted")
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            parsed = {}
        evidence = parsed.get("evidence", [])
        result = {
            "diagnosis": str(parsed.get("diagnosis") or "unknown"),
            "evidence": [str(item) for item in evidence] if isinstance(evidence, list) else [],
        }
        self.cache[key] = result
        save_json(self.cache_path, self.cache)
        return result


def evaluate_inference_attack(
    cases: list[dict[str, Any]], attacker: Attacker
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Each case needs document_id, dataset, diagnosis and versions {name: text}."""
    prior = attacker(NO_RECORD)
    rows = []
    for case in cases:
        for version, text in case["versions"].items():
            guess = attacker(text)
            rows.append(
                {
                    "document_id": case["document_id"],
                    "dataset": case["dataset"],
                    "version": version,
                    "diagnosis": case["diagnosis"],
                    "literal_diagnosis_visible": int(
                        diagnosis_recovered(case["diagnosis"], text)
                    ),
                    "attacker_guess": guess["diagnosis"],
                    "recovered": int(diagnosis_recovered(case["diagnosis"], guess["diagnosis"])),
                    "evidence": "; ".join(guess["evidence"]),
                }
            )

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["version"]].append(row)
    versions = {}
    for version, items in grouped.items():
        by_dataset = defaultdict(list)
        for item in items:
            by_dataset[item["dataset"]].append(item["recovered"])
        versions[version] = {
            "records": len(items),
            "literal_visible_rate": round(
                statistics.fmean(item["literal_diagnosis_visible"] for item in items), 4
            ),
            "recovery_rate": round(statistics.fmean(item["recovered"] for item in items), 4),
            "recovery_by_dataset": {
                name: round(statistics.fmean(values), 4)
                for name, values in sorted(by_dataset.items())
            },
        }
    prior_rate = round(
        statistics.fmean(
            diagnosis_recovered(case["diagnosis"], prior["diagnosis"]) for case in cases
        ),
        4,
    ) if cases else 0.0
    summary = {
        "benchmark": {
            "records": len(cases),
            "datasets": sorted({case["dataset"] for case in cases}),
            "scoring": "conservative lexical match; synonyms count as misses",
        },
        "prior": {"guess": prior["diagnosis"], "recovery_rate": prior_rate},
        "versions": versions,
    }
    return rows, summary


def inference_table(summary: dict[str, Any]) -> list[str]:
    datasets = summary["benchmark"]["datasets"]
    lines = [
        "| Representation | Diagnosis string visible | Attacker recovered (all) | "
        + " | ".join(f"Recovered ({name})" for name in datasets) + " |",
        "|---|---:|---:|" + "---:|" * len(datasets),
    ]
    for version, item in summary["versions"].items():
        by_dataset = " | ".join(
            f"{item['recovery_by_dataset'].get(name, 0.0) * 100:.2f}%" for name in datasets
        )
        lines.append(
            f"| {version} | {item['literal_visible_rate'] * 100:.2f}% | "
            f"{item['recovery_rate'] * 100:.2f}% | {by_dataset} |"
        )
    return lines


INFERENCE_NOTES = (
    "`fixed_low` shows every diagnosis, so its recovery rate is the ceiling this scorer "
    "can detect. Scoring is a conservative lexical match (GERD does not match "
    "gastroesophageal reflux), and ChatDoctor labels come from a keyword heuristic that "
    "is sometimes wrong, so held-out records give the cleaner estimate and all recovery "
    "rates are lower bounds on inferential leakage."
)


def render_inference_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Attacker-LLM Diagnosis Inference",
        "",
        f"Records: {summary['benchmark']['records']} medical records "
        f"({', '.join(summary['benchmark']['datasets'])}). An LLM adversary reads each "
        "released representation and guesses the hidden diagnosis.",
        "",
        *inference_table(summary),
        "",
        f"No-record prior: the attacker guessed `{summary['prior']['guess']}`, which matches "
        f"{summary['prior']['recovery_rate'] * 100:.2f}% of records by chance.",
        "",
        INFERENCE_NOTES,
        "",
    ]
    return "\n".join(lines)

"""Adapter and evaluation for the external Agentic Prompt-Injection 5K split."""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .challenge_evaluation import binary_metrics
from .integrity import PromptInjectionGuard


DATASET_ID = "3nesdeniz/agentic-prompt-injection-5k"
DATASET_PAGE = f"https://huggingface.co/datasets/{DATASET_ID}"
ROWS_ENDPOINT = "https://datasets-server.huggingface.co/rows"


def fetch_external_rows(
    split: str = "test",
    page_size: int = 100,
    timeout: int = 30,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not 1 <= page_size <= 100:
        raise ValueError("page_size must be from 1 to 100.")
    rows: list[dict[str, Any]] = []
    offset = 0
    total = None
    revision = "unknown"
    while total is None or offset < total:
        parameters = urlencode(
            {
                "dataset": DATASET_ID,
                "config": "default",
                "split": split,
                "offset": offset,
                "length": page_size,
            }
        )
        request = Request(
            f"{ROWS_ENDPOINT}?{parameters}",
            headers={"User-Agent": "ppr-lab-project/1.0"},
        )
        with urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
            revision = response.headers.get("x-revision", revision)
        batch = [
            {"row_idx": item["row_idx"], **item["row"]}
            for item in payload.get("rows", [])
        ]
        rows.extend(batch)
        total = int(payload.get("num_rows_total", len(rows)))
        if not batch:
            break
        offset += len(batch)
    source = {
        "dataset_id": DATASET_ID,
        "dataset_page": DATASET_PAGE,
        "revision": revision,
        "split": split,
        "rows": len(rows),
        "reported_total": total,
        "license": "CC-BY-4.0",
    }
    return rows, source


def evaluate_external_rows(
    rows: list[dict[str, Any]], guard: Any
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for item in rows:
        assessment = guard.assess(str(item["text"]))
        attack = int(item["label"]) == 1
        results.append(
            {
                "row_idx": item.get("row_idx", item.get("id")),
                "attack_family": item.get("attack_family", "unknown"),
                "technique": item.get("technique", "unknown"),
                "attack": int(attack),
                "detected": int(assessment.detected),
                "correct": int(attack == assessment.detected),
                "score": assessment.score,
                "detected_categories": "|".join(assessment.categories),
            }
        )
    family_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        family_groups[str(row["attack_family"])].append(row)
    summary = {
        "overall": binary_metrics(results),
        "per_attack_family": {
            family: binary_metrics(family_rows)
            for family, family_rows in sorted(family_groups.items())
        },
    }
    return results, summary


def render_external_report(summary: dict[str, Any]) -> str:
    source = summary["benchmark"]
    baseline = summary["phrase_baseline"]["overall"]
    structural = summary["compositional_guard"]["overall"]
    learned = summary.get("learned_tfidf_centroid", {}).get("overall")
    lines = [
        "# External Agentic Prompt-Injection Evaluation",
        "",
        f"Dataset: [{source['dataset_id']}]({source['dataset_page']})  ",
        f"Revision: `{source['revision']}`  ",
        f"Split: `{source['split']}` ({source['rows']} rows)  ",
        f"License: {source['license']}",
        "",
        "| Detector | Recall | Precision | False-positive rate | Accuracy |",
        "|---|---:|---:|---:|---:|",
        f"| Exact-phrase baseline | {baseline['attack_recall']:.1%} | {baseline['attack_precision']:.1%} | {baseline['benign_false_positive_rate']:.1%} | {baseline['accuracy']:.1%} |",
        f"| Compositional guard | {structural['attack_recall']:.1%} | {structural['attack_precision']:.1%} | {structural['benign_false_positive_rate']:.1%} | {structural['accuracy']:.1%} |",
    ]
    if learned is not None:
        lines.append(
            f"| Learned char TF-IDF centroid | {learned['attack_recall']:.1%} | {learned['attack_precision']:.1%} | {learned['benign_false_positive_rate']:.1%} | {learned['accuracy']:.1%} |"
        )
    lines.extend(
        [
            "",
            "The learned baseline is fitted only on the training split and selects its threshold only on validation; the scenario-isolated test split remains evaluation-only.",
            "",
            "This is an external synthetic benchmark. Results apply only to the downloaded revision and do not constitute a security guarantee.",
            "",
        ]
    )
    return "\n".join(lines)

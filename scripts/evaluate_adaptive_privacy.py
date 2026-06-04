"""Compare fixed-low, fixed-high and adaptive privacy policies."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from adaptive_privacy.config import load_config
from adaptive_privacy.detector import PrivacyDetector
from adaptive_privacy.io import load_json
from adaptive_privacy.pipeline import AdaptivePrivacyPipeline
from adaptive_privacy.retriever import tokenize
from adaptive_privacy.tier_builder import TierBuilder


def value_retention(raw_text: str, protected_text: str, category: str | None) -> float:
    entities = PrivacyDetector().detect(raw_text)
    selected = [item for item in entities if category is None or item.category == category]
    if not selected:
        return 0.0
    protected_lower = protected_text.lower()
    retained = sum(item.value.lower() in protected_lower for item in selected)
    return retained / len(selected)


def utility_retention(raw_text: str, protected_text: str) -> float:
    entities = PrivacyDetector().detect(raw_text)
    safe_parts = []
    cursor = 0
    for entity in entities:
        safe_parts.append(raw_text[cursor : entity.start])
        cursor = entity.end
    safe_parts.append(raw_text[cursor:])
    reference_tokens = set(tokenize(" ".join(safe_parts)))
    if not reference_tokens:
        return 1.0
    protected_tokens = set(tokenize(protected_text))
    return len(reference_tokens & protected_tokens) / len(reference_tokens)


def evaluate(args: argparse.Namespace) -> tuple[list[dict], dict]:
    raw_documents = load_json(args.documents)
    raw_by_id = {
        str(item.get("document_id", item.get("id"))): item["text"]
        for item in raw_documents
    }
    queries = load_json(args.queries)
    tiered = TierBuilder().build_many(raw_documents)
    pipeline = AdaptivePrivacyPipeline(tiered, config=load_config(args.config))

    rows = []
    modes = ("fixed_low", "fixed_high", "adaptive")
    for query_case in queries:
        for mode in modes:
            result = pipeline.process(
                query_case["query"], query_case.get("role", "unknown"), mode=mode
            )
            direct_rates = []
            all_rates = []
            utility_rates = []
            for decision, protected in zip(
                result.policy_decisions, result.protected_contexts
            ):
                raw = raw_by_id[decision.document_id]
                direct_rates.append(value_retention(raw, protected, "direct"))
                all_rates.append(value_retention(raw, protected, None))
                utility_rates.append(utility_retention(raw, protected))

            rows.append(
                {
                    "query_id": query_case["query_id"],
                    "mode": mode,
                    "role": query_case.get("role", "unknown"),
                    "expected_query_risk": query_case["expected_query_risk"],
                    "predicted_query_risk": result.query_decision.level.value,
                    "risk_correct": int(
                        result.query_decision.level.value
                        == query_case["expected_query_risk"]
                    ),
                    "documents_retrieved": len(result.policy_decisions),
                    "direct_identifier_retention": round(
                        statistics.fmean(direct_rates) if direct_rates else 0.0, 4
                    ),
                    "all_sensitive_value_retention": round(
                        statistics.fmean(all_rates) if all_rates else 0.0, 4
                    ),
                    "utility_retention": round(
                        statistics.fmean(utility_rates) if utility_rates else 0.0, 4
                    ),
                    "latency_ms": round(result.latency_ms, 4),
                    "blocked": int(result.blocked),
                }
            )

    summary: dict[str, dict] = {}
    for mode in modes:
        mode_rows = [row for row in rows if row["mode"] == mode]
        summary[mode] = {
            "queries": len(mode_rows),
            "query_risk_accuracy": round(
                statistics.fmean(row["risk_correct"] for row in mode_rows), 4
            ),
            "direct_identifier_retention": round(
                statistics.fmean(
                    row["direct_identifier_retention"] for row in mode_rows
                ),
                4,
            ),
            "all_sensitive_value_retention": round(
                statistics.fmean(
                    row["all_sensitive_value_retention"] for row in mode_rows
                ),
                4,
            ),
            "utility_retention": round(
                statistics.fmean(row["utility_retention"] for row in mode_rows), 4
            ),
            "mean_latency_ms": round(
                statistics.fmean(row["latency_ms"] for row in mode_rows), 4
            ),
        }
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--documents",
        type=Path,
        default=PROJECT_ROOT / "examples" / "adaptive_privacy" / "documents.json",
    )
    parser.add_argument(
        "--queries",
        type=Path,
        default=PROJECT_ROOT / "examples" / "adaptive_privacy" / "queries.json",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "adaptive_privacy.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "adaptive_privacy",
    )
    args = parser.parse_args()

    rows, summary = evaluate(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "evaluation.csv"
    json_path = args.output_dir / "summary.json"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Detailed results -> {csv_path}")


if __name__ == "__main__":
    main()

"""Ground-truth evaluation that does not reuse the production PII detector."""

from __future__ import annotations

import re
import statistics
from collections import Counter, defaultdict
from typing import Any

from .pipeline import AdaptivePrivacyPipeline


def strip_placeholders(text: str) -> str:
    return re.sub(r"\[[^\]]+\]", " ", text)


def normalize(text: str) -> str:
    return " ".join(text.casefold().split())


def literal_present(value: str, text: str) -> bool:
    candidate = normalize(strip_placeholders(text))
    expected = normalize(value)
    if not expected:
        return False
    return re.search(rf"(?<!\w){re.escape(expected)}(?!\w)", candidate) is not None


def evaluate_ground_truth(
    documents: list[dict[str, Any]],
    queries: list[dict[str, Any]],
    pipeline: AdaptivePrivacyPipeline,
    modes: tuple[str, ...] = ("fixed_low", "fixed_high", "adaptive"),
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    documents_by_id = {str(item["document_id"]): item for item in documents}
    rows: list[dict[str, Any]] = []

    for query_case in queries:
        for mode in modes:
            result = pipeline.process(
                query_case["query"],
                role=query_case.get("role", "unknown"),
                role_verified=bool(query_case.get("role_verified", False)),
                mode=mode,
            )
            retrieved_ids = [item.document_id for item in result.policy_decisions]
            privacy_totals: Counter[str] = Counter()
            privacy_retained: Counter[str] = Counter()
            utility_total = 0
            utility_retained = 0

            for document_id, protected_context in zip(
                retrieved_ids, result.protected_contexts
            ):
                ground_truth = documents_by_id[document_id]["ground_truth"]
                for item in ground_truth["privacy_values"]:
                    category = item["category"]
                    privacy_totals[category] += 1
                    if literal_present(item["value"], protected_context):
                        privacy_retained[category] += 1
                for term in ground_truth["utility_terms"]:
                    utility_total += 1
                    if literal_present(term, protected_context):
                        utility_retained += 1

            def retention(category: str) -> float:
                total = privacy_totals[category]
                return privacy_retained[category] / total if total else 0.0

            rows.append(
                {
                    "query_id": query_case["query_id"],
                    "mode": mode,
                    "role": query_case.get("role", "unknown"),
                    "role_verified": int(query_case.get("role_verified", False)),
                    "expected_query_risk": query_case["expected_query_risk"],
                    "predicted_query_risk": result.query_decision.level.value,
                    "risk_correct": int(
                        result.query_decision.level.value
                        == query_case["expected_query_risk"]
                    ),
                    "retrieval_hit": int(
                        query_case["relevant_document_id"] in retrieved_ids
                    ),
                    "retrieved_documents": len(retrieved_ids),
                    "selected_tiers": "|".join(
                        item.selected_tier.value for item in result.policy_decisions
                    ),
                    "direct_retention": round(retention("direct"), 4),
                    "quasi_retention": round(retention("quasi"), 4),
                    "sensitive_retention": round(retention("sensitive"), 4),
                    "utility_retention": round(
                        utility_retained / utility_total if utility_total else 1.0,
                        4,
                    ),
                    "latency_ms": round(result.latency_ms, 4),
                    "blocked": int(result.blocked),
                }
            )

    summary: dict[str, Any] = {
        "benchmark": {
            "documents": len(documents),
            "queries": len(queries),
            "ground_truth_source": "explicit benchmark annotations",
        },
        "modes": {},
    }
    for mode in modes:
        mode_rows = [row for row in rows if row["mode"] == mode]
        tier_counts = Counter(
            tier
            for row in mode_rows
            for tier in str(row["selected_tiers"]).split("|")
            if tier
        )
        by_expected_risk: dict[str, dict[str, float]] = {}
        for risk in ("low", "medium", "high"):
            risk_rows = [row for row in mode_rows if row["expected_query_risk"] == risk]
            by_expected_risk[risk] = {
                "queries": len(risk_rows),
                "direct_retention": round(
                    statistics.fmean(row["direct_retention"] for row in risk_rows), 4
                ),
                "quasi_retention": round(
                    statistics.fmean(row["quasi_retention"] for row in risk_rows), 4
                ),
                "sensitive_retention": round(
                    statistics.fmean(row["sensitive_retention"] for row in risk_rows),
                    4,
                ),
                "utility_retention": round(
                    statistics.fmean(row["utility_retention"] for row in risk_rows), 4
                ),
            }
        summary["modes"][mode] = {
            "query_risk_accuracy": round(
                statistics.fmean(row["risk_correct"] for row in mode_rows), 4
            ),
            "retrieval_hit_rate": round(
                statistics.fmean(row["retrieval_hit"] for row in mode_rows), 4
            ),
            "direct_retention": round(
                statistics.fmean(row["direct_retention"] for row in mode_rows), 4
            ),
            "quasi_retention": round(
                statistics.fmean(row["quasi_retention"] for row in mode_rows), 4
            ),
            "sensitive_retention": round(
                statistics.fmean(row["sensitive_retention"] for row in mode_rows), 4
            ),
            "utility_retention": round(
                statistics.fmean(row["utility_retention"] for row in mode_rows), 4
            ),
            "mean_latency_ms": round(
                statistics.fmean(row["latency_ms"] for row in mode_rows), 4
            ),
            "blocked_answers": sum(row["blocked"] for row in mode_rows),
            "selected_tier_counts": dict(sorted(tier_counts.items())),
            "by_expected_query_risk": by_expected_risk,
        }
    return rows, summary


def render_markdown_report(summary: dict[str, Any]) -> str:
    benchmark = summary["benchmark"]
    lines = [
        "# Held-out Adaptive Privacy Evaluation",
        "",
        f"Documents: {benchmark['documents']}  ",
        f"Queries: {benchmark['queries']}  ",
        "Ground truth: explicit annotations independent of the production detector",
        "",
        "| Mode | Risk accuracy | Retrieval hit | Direct retention | Quasi retention | Sensitive retention | Utility retention | Mean latency (ms) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode, metrics in summary["modes"].items():
        lines.append(
            "| {mode} | {risk:.1%} | {hit:.1%} | {direct:.1%} | {quasi:.1%} | "
            "{sensitive:.1%} | {utility:.1%} | {latency:.3f} |".format(
                mode=mode,
                risk=metrics["query_risk_accuracy"],
                hit=metrics["retrieval_hit_rate"],
                direct=metrics["direct_retention"],
                quasi=metrics["quasi_retention"],
                sensitive=metrics["sensitive_retention"],
                utility=metrics["utility_retention"],
                latency=metrics["mean_latency_ms"],
            )
        )
    lines.extend(
        [
            "",
            "## Adaptive policy by expected query risk",
            "",
            "| Expected risk | Queries | Direct retention | Quasi retention | Sensitive retention | Utility retention |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    adaptive_breakdown = summary["modes"]["adaptive"]["by_expected_query_risk"]
    for risk in ("low", "medium", "high"):
        metrics = adaptive_breakdown[risk]
        lines.append(
            "| {risk} | {queries} | {direct:.1%} | {quasi:.1%} | {sensitive:.1%} | {utility:.1%} |".format(
                risk=risk,
                queries=metrics["queries"],
                direct=metrics["direct_retention"],
                quasi=metrics["quasi_retention"],
                sensitive=metrics["sensitive_retention"],
                utility=metrics["utility_retention"],
            )
        )
    lines.extend(
        [
            "",
            "Retention is the fraction of annotated private values still visible; lower is better. Utility retention is the fraction of annotated task-useful phrases preserved; higher is better.",
            "",
            "This deterministic synthetic benchmark is suitable for lab comparison and regression testing, not as a formal privacy guarantee.",
        ]
    )
    return "\n".join(lines) + "\n"

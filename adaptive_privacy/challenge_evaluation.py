"""Evaluation for the frozen multilingual prompt-injection challenge set."""

from __future__ import annotations

import hashlib
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from .risk_classifier import QueryRiskClassifier


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_challenge(challenge: dict[str, Any]) -> None:
    cases = challenge.get("cases", [])
    if not cases:
        raise ValueError("Challenge set contains no cases.")
    case_ids = [str(item["case_id"]) for item in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Challenge case IDs must be unique.")
    for language in challenge.get("languages", []):
        language_cases = [item for item in cases if item["language"] == language]
        labels = {bool(item["attack"]) for item in language_cases}
        if labels != {False, True}:
            raise ValueError(f"Language {language!r} must contain both labels.")


def binary_metrics(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    true_positive = sum(row["attack"] and row["detected"] for row in rows)
    false_negative = sum(row["attack"] and not row["detected"] for row in rows)
    false_positive = sum(not row["attack"] and row["detected"] for row in rows)
    true_negative = sum(not row["attack"] and not row["detected"] for row in rows)
    attacks = true_positive + false_negative
    benign = true_negative + false_positive
    predicted_attacks = true_positive + false_positive
    total = len(rows)
    return {
        "cases": total,
        "attacks": attacks,
        "benign": benign,
        "true_positive": true_positive,
        "false_negative": false_negative,
        "false_positive": false_positive,
        "true_negative": true_negative,
        "accuracy": round((true_positive + true_negative) / total, 4),
        "attack_recall": round(true_positive / attacks if attacks else 0.0, 4),
        "attack_precision": round(
            true_positive / predicted_attacks if predicted_attacks else 0.0, 4
        ),
        "benign_false_positive_rate": round(
            false_positive / benign if benign else 0.0, 4
        ),
    }


def evaluate_injection_challenge(
    challenge: dict[str, Any],
    classifier: QueryRiskClassifier,
    source_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    validate_challenge(challenge)
    rows: list[dict[str, Any]] = []
    for item in challenge["cases"]:
        assessment = classifier.injection_guard.assess(str(item["query"]))
        attack = bool(item["attack"])
        detected = assessment.detected
        rows.append(
            {
                "case_id": item["case_id"],
                "language": item["language"],
                "category": item["category"],
                "attack": int(attack),
                "detected": int(detected),
                "correct": int(attack == detected),
                "score": assessment.score,
                "detected_categories": "|".join(assessment.categories),
            }
        )

    per_language = {}
    for language in challenge["languages"]:
        per_language[language] = binary_metrics(
            [row for row in rows if row["language"] == language]
        )
    category_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        category_groups[str(row["category"])].append(row)
    per_category = {
        category: binary_metrics(category_rows)
        for category, category_rows in sorted(category_groups.items())
    }
    summary = {
        "benchmark": {
            "name": challenge["benchmark"],
            "version": challenge["version"],
            "frozen_on": challenge["frozen_on"],
            "source": source_path.as_posix(),
            "sha256": file_sha256(source_path),
            "languages": challenge["languages"],
        },
        "overall": binary_metrics(rows),
        "per_language": per_language,
        "per_category": per_category,
        "mean_detection_score": round(
            statistics.fmean(row["score"] for row in rows), 4
        ),
    }
    return rows, summary


def render_challenge_report(summary: dict[str, Any]) -> str:
    benchmark = summary["benchmark"]
    overall = summary["overall"]
    lines = [
        "# Frozen Multilingual Prompt-Injection Challenge",
        "",
        f"Version: `{benchmark['version']}`  ",
        f"Frozen: {benchmark['frozen_on']}  ",
        f"SHA-256: `{benchmark['sha256']}`",
        "",
        "| Metric | Result |",
        "|---|---:|",
        f"| Cases | {overall['cases']} |",
        f"| Attack recall | {overall['attack_recall']:.1%} |",
        f"| Attack precision | {overall['attack_precision']:.1%} |",
        f"| Benign false-positive rate | {overall['benign_false_positive_rate']:.1%} |",
        f"| Accuracy | {overall['accuracy']:.1%} |",
        "",
    ]
    if summary.get("baseline_phrase_only"):
        baseline = summary["baseline_phrase_only"]["overall"]
        lines.extend(
            [
                "## Phrase baseline comparison",
                "",
                "| Detector | Attack recall | Precision | False-positive rate | Accuracy |",
                "|---|---:|---:|---:|---:|",
                f"| Exact-phrase baseline | {baseline['attack_recall']:.1%} | {baseline['attack_precision']:.1%} | {baseline['benign_false_positive_rate']:.1%} | {baseline['accuracy']:.1%} |",
                f"| Compositional multilingual guard | {overall['attack_recall']:.1%} | {overall['attack_precision']:.1%} | {overall['benign_false_positive_rate']:.1%} | {overall['accuracy']:.1%} |",
                "",
            ]
        )
    lines.extend(
        [
        "## Per-language performance",
        "",
        "| Language | Cases | Recall | Precision | False-positive rate | Accuracy |",
        "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for language, metrics in summary["per_language"].items():
        lines.append(
            f"| {language} | {metrics['cases']} | {metrics['attack_recall']:.1%} | "
            f"{metrics['attack_precision']:.1%} | "
            f"{metrics['benign_false_positive_rate']:.1%} | {metrics['accuracy']:.1%} |"
        )
    lines.extend(
        [
            "",
            "This fixed challenge is a small regression benchmark. It does not establish multilingual robustness, resistance to adaptive attacks, or real-world prevalence.",
            "",
        ]
    )
    return "\n".join(lines)

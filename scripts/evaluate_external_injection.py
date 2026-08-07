"""Download and evaluate the external Agentic Prompt-Injection 5K test split."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from adaptive_privacy.external_injection import (
    DATASET_PAGE,
    evaluate_external_rows,
    fetch_external_rows,
    render_external_report,
)
from adaptive_privacy.integrity import PromptInjectionGuard
from adaptive_privacy.io import load_json, save_json
from adaptive_privacy.learned_detector import TfidfCentroidInjectionDetector


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache",
        type=Path,
        default=PROJECT_ROOT / "data" / "external" / "agentic_injection_test.json",
    )
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument(
        "--train-cache",
        type=Path,
        default=PROJECT_ROOT / "data" / "external" / "agentic_injection_train.json",
    )
    parser.add_argument(
        "--validation-cache",
        type=Path,
        default=PROJECT_ROOT / "data" / "external" / "agentic_injection_validation.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "adaptive_privacy" / "external_injection",
    )
    args = parser.parse_args()

    def load_split(path: Path, split: str):
        if path.exists() and not args.refresh:
            cached = load_json(path)
            print(f"Using cached external {split} split -> {path}")
            return cached["rows"], cached["source"]
        split_rows, split_source = fetch_external_rows(split=split)
        save_json(path, {"source": split_source, "rows": split_rows})
        print(f"Downloaded {len(split_rows)} external {split} rows -> {path}")
        return split_rows, split_source

    rows, source = load_split(args.cache, "test")
    train_rows, train_source = load_split(args.train_cache, "train")
    validation_rows, validation_source = load_split(
        args.validation_cache, "validation"
    )

    baseline_rows, baseline = evaluate_external_rows(
        rows, PromptInjectionGuard(structural=False)
    )
    structural_rows, structural = evaluate_external_rows(
        rows, PromptInjectionGuard(structural=True)
    )
    learned_detector = TfidfCentroidInjectionDetector()
    learned_detector.fit(
        [str(item["text"]) for item in train_rows],
        [int(item["label"]) for item in train_rows],
    )
    learned_detector.tune_threshold(
        [str(item["text"]) for item in validation_rows],
        [int(item["label"]) for item in validation_rows],
    )
    learned_rows, learned = evaluate_external_rows(rows, learned_detector)
    summary = {
        "benchmark": {
            **source,
            "dataset_page": DATASET_PAGE,
            "note": "External scenario-isolated test split; raw rows excluded from submission archive.",
        },
        "phrase_baseline": baseline,
        "compositional_guard": structural,
        "learned_tfidf_centroid": learned,
        "learned_model": {
            "training_rows": len(train_rows),
            "validation_rows": len(validation_rows),
            "test_rows": len(rows),
            "vocabulary_features": len(learned_detector.vocabulary),
            "decision_threshold": round(learned_detector.threshold, 6),
            "train_revision": train_source.get("revision", "unknown"),
            "validation_revision": validation_source.get("revision", "unknown"),
            "test_revision": source.get("revision", "unknown"),
            "test_labels_used_for_training_or_threshold": False,
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_json(args.output_dir / "summary.json", summary)
    for filename, result_rows in (
        ("phrase_baseline.csv", baseline_rows),
        ("compositional_guard.csv", structural_rows),
        ("learned_tfidf_centroid.csv", learned_rows),
    ):
        with (args.output_dir / filename).open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(result_rows[0]))
            writer.writeheader()
            writer.writerows(result_rows)
    (args.output_dir / "report.md").write_text(
        render_external_report(summary), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Report -> {args.output_dir / 'report.md'}")


if __name__ == "__main__":
    main()

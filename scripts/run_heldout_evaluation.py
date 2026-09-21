"""Generate and evaluate the independently annotated held-out benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from adaptive_privacy.benchmark import generate_heldout_benchmark
from adaptive_privacy.config import load_config
from adaptive_privacy.evaluation import (
    evaluate_ground_truth,
    reidentification_ablation,
    render_markdown_report,
)
from adaptive_privacy.io import save_json
from adaptive_privacy.pipeline import AdaptivePrivacyPipeline
from adaptive_privacy.tier_builder import TierBuilder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "adaptive_privacy.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "adaptive_privacy" / "heldout",
    )
    args = parser.parse_args()

    documents, queries = generate_heldout_benchmark()
    tiered_documents = TierBuilder().build_many(documents)
    config = load_config(args.config)
    pipeline = AdaptivePrivacyPipeline(tiered_documents, config=config)
    rows, summary = evaluate_ground_truth(
        documents, queries, pipeline,
        ablations=reidentification_ablation(tiered_documents, config),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_json(args.output_dir / "documents.json", documents)
    save_json(args.output_dir / "queries.json", queries)
    save_json(args.output_dir / "summary.json", summary)
    with (args.output_dir / "evaluation.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = render_markdown_report(summary)
    (args.output_dir / "report.md").write_text(report, encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Report -> {args.output_dir / 'report.md'}")


if __name__ == "__main__":
    main()

"""Fetch 24 official ChatDoctor rows, inject controlled PII, and evaluate."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from adaptive_privacy.chatdoctor import (
    DATASET_PAGE,
    fetch_chatdoctor_rows,
    prepare_chatdoctor_benchmark,
)
from adaptive_privacy.config import load_config
from adaptive_privacy.evaluation import evaluate_ground_truth, render_markdown_report
from adaptive_privacy.io import load_json, save_json
from adaptive_privacy.pipeline import AdaptivePrivacyPipeline
from adaptive_privacy.tier_builder import TierBuilder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument(
        "--cache",
        type=Path,
        default=PROJECT_ROOT / "data" / "external" / "chatdoctor_sample.json",
    )
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "adaptive_privacy.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "adaptive_privacy" / "chatdoctor",
    )
    args = parser.parse_args()

    if args.cache.exists() and not args.refresh:
        cached = load_json(args.cache)
        rows = cached["rows"]
        source = cached["source"]
        print(f"Using cached ChatDoctor sample -> {args.cache}")
    else:
        rows, source = fetch_chatdoctor_rows(args.limit, args.offset)
        save_json(args.cache, {"source": source, "rows": rows})
        print(f"Downloaded {len(rows)} ChatDoctor rows -> {args.cache}")

    documents, queries = prepare_chatdoctor_benchmark(rows, source)
    pipeline = AdaptivePrivacyPipeline(
        TierBuilder().build_many(documents), config=load_config(args.config)
    )
    result_rows, summary = evaluate_ground_truth(documents, queries, pipeline)
    summary["benchmark"].update(
        {
            "name": "ChatDoctor sample with controlled synthetic PII",
            "source": DATASET_PAGE,
            "source_revision": source.get("revision", "unknown"),
            "source_rows": len(rows),
            "note": "Original records are de-identified; PII fields are synthetic.",
        }
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_json(args.output_dir / "documents.json", documents)
    save_json(args.output_dir / "queries.json", queries)
    save_json(args.output_dir / "summary.json", summary)
    with (args.output_dir / "evaluation.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result_rows[0]))
        writer.writeheader()
        writer.writerows(result_rows)
    (args.output_dir / "report.md").write_text(
        render_markdown_report(summary), encoding="utf-8"
    )

    print(json.dumps(summary, indent=2))
    print(f"Report -> {args.output_dir / 'report.md'}")


if __name__ == "__main__":
    main()

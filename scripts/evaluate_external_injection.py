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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache",
        type=Path,
        default=PROJECT_ROOT / "data" / "external" / "agentic_injection_test.json",
    )
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "adaptive_privacy" / "external_injection",
    )
    args = parser.parse_args()

    if args.cache.exists() and not args.refresh:
        cached = load_json(args.cache)
        rows = cached["rows"]
        source = cached["source"]
        print(f"Using cached external split -> {args.cache}")
    else:
        rows, source = fetch_external_rows()
        save_json(args.cache, {"source": source, "rows": rows})
        print(f"Downloaded {len(rows)} external test rows -> {args.cache}")

    baseline_rows, baseline = evaluate_external_rows(
        rows, PromptInjectionGuard(structural=False)
    )
    structural_rows, structural = evaluate_external_rows(
        rows, PromptInjectionGuard(structural=True)
    )
    summary = {
        "benchmark": {
            **source,
            "dataset_page": DATASET_PAGE,
            "note": "External scenario-isolated test split; raw rows excluded from submission archive.",
        },
        "phrase_baseline": baseline,
        "compositional_guard": structural,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_json(args.output_dir / "summary.json", summary)
    for filename, result_rows in (
        ("phrase_baseline.csv", baseline_rows),
        ("compositional_guard.csv", structural_rows),
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

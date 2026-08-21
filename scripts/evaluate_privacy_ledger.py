"""Evaluate purpose binding and disclosure accounting across fresh sessions."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from adaptive_privacy.config import load_config
from adaptive_privacy.io import load_json, save_json
from adaptive_privacy.ledger_evaluation import (
    evaluate_privacy_ledger,
    render_ledger_report,
)
from adaptive_privacy.tier_builder import TierBuilder


def main() -> None:
    documents = TierBuilder().build_many(
        load_json(PROJECT_ROOT / "examples" / "adaptive_privacy" / "documents.json")
    )
    output_dir = PROJECT_ROOT / "results" / "adaptive_privacy" / "ledger"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, summary = evaluate_privacy_ledger(
        documents,
        load_config(),
        output_dir / "evaluation_ledger.json",
    )
    save_json(output_dir / "summary.json", summary)
    with (output_dir / "evaluation.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "report.md").write_text(
        render_ledger_report(summary), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(f"Report -> {output_dir / 'report.md'}")


if __name__ == "__main__":
    main()

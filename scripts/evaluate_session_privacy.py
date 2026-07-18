"""Run controlled benign and multi-turn extraction scenarios."""

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
from adaptive_privacy.pipeline import AdaptivePrivacyPipeline
from adaptive_privacy.session_evaluation import (
    evaluate_ablations,
    evaluate_sessions,
    render_session_report,
)
from adaptive_privacy.session_pipeline import SessionAwarePrivacyPipeline
from adaptive_privacy.tier_builder import TierBuilder


def main() -> None:
    documents = TierBuilder().build_many(
        load_json(PROJECT_ROOT / "examples" / "adaptive_privacy" / "documents.json")
    )
    config = load_config()
    pipeline = SessionAwarePrivacyPipeline(
        AdaptivePrivacyPipeline(documents, config=config)
    )
    rows, summary = evaluate_sessions(pipeline)
    summary["ablations"] = evaluate_ablations(documents, config)
    output_dir = PROJECT_ROOT / "results" / "adaptive_privacy" / "sessions"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "summary.json", summary)
    with (output_dir / "evaluation.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "ablation.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(summary["ablations"][0])
        )
        writer.writeheader()
        writer.writerows(summary["ablations"])
    (output_dir / "report.md").write_text(
        render_session_report(summary), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(f"Report -> {output_dir / 'report.md'}")


if __name__ == "__main__":
    main()

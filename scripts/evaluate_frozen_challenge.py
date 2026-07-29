"""Evaluate the fixed multilingual prompt-injection challenge set."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from adaptive_privacy.challenge_evaluation import (
    evaluate_injection_challenge,
    render_challenge_report,
)
from adaptive_privacy.config import load_config
from adaptive_privacy.io import load_json, save_json
from adaptive_privacy.risk_classifier import QueryRiskClassifier


def main() -> None:
    source_path = (
        PROJECT_ROOT
        / "examples"
        / "adaptive_privacy"
        / "frozen_injection_challenge.json"
    )
    challenge = load_json(source_path)
    rows, summary = evaluate_injection_challenge(
        challenge, QueryRiskClassifier(load_config()), source_path
    )
    output_dir = PROJECT_ROOT / "results" / "adaptive_privacy" / "frozen_challenge"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "summary.json", summary)
    with (output_dir / "evaluation.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "report.md").write_text(
        render_challenge_report(summary), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Report -> {output_dir / 'report.md'}")


if __name__ == "__main__":
    main()

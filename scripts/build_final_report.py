"""Combine the controlled and ChatDoctor experiments into a lab report."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from adaptive_privacy.io import load_json
from adaptive_privacy.reporting import generate_final_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--controlled-summary",
        type=Path,
        default=(
            PROJECT_ROOT
            / "results"
            / "adaptive_privacy"
            / "heldout"
            / "summary.json"
        ),
    )
    parser.add_argument(
        "--chatdoctor-summary",
        type=Path,
        default=(
            PROJECT_ROOT
            / "results"
            / "adaptive_privacy"
            / "chatdoctor"
            / "summary.json"
        ),
    )
    parser.add_argument(
        "--session-summary",
        type=Path,
        default=(
            PROJECT_ROOT
            / "results"
            / "adaptive_privacy"
            / "sessions"
            / "summary.json"
        ),
    )
    parser.add_argument(
        "--challenge-summary",
        type=Path,
        default=(
            PROJECT_ROOT
            / "results"
            / "adaptive_privacy"
            / "frozen_challenge"
            / "summary.json"
        ),
    )
    parser.add_argument(
        "--external-injection-summary",
        type=Path,
        default=(
            PROJECT_ROOT
            / "results"
            / "adaptive_privacy"
            / "external_injection"
            / "summary.json"
        ),
    )
    parser.add_argument(
        "--ledger-summary",
        type=Path,
        default=(
            PROJECT_ROOT
            / "results"
            / "adaptive_privacy"
            / "ledger"
            / "summary.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            PROJECT_ROOT
            / "results"
            / "adaptive_privacy"
            / "final_lab_report.md"
        ),
    )
    args = parser.parse_args()

    missing = [
        path
        for path in (
            args.controlled_summary,
            args.chatdoctor_summary,
            args.session_summary,
            args.challenge_summary,
            args.external_injection_summary,
            args.ledger_summary,
        )
        if not path.exists()
    ]
    if missing:
        paths = ", ".join(str(path) for path in missing)
        raise SystemExit(
            f"Missing evaluation summaries: {paths}. Run both evaluation scripts first."
        )

    inference_path = PROJECT_ROOT / "results" / "adaptive_privacy" / "inference_attack" / "summary.json"
    report = generate_final_report(
        load_json(args.controlled_summary),
        load_json(args.chatdoctor_summary),
        load_json(args.session_summary),
        load_json(args.challenge_summary),
        load_json(args.external_injection_summary),
        load_json(args.ledger_summary),
        # Optional: needs a Groq key, so the report builds without it.
        load_json(inference_path) if inference_path.exists() else None,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"Final lab report -> {args.output}")


if __name__ == "__main__":
    main()

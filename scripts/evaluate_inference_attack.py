"""Measure diagnosis inference by an attacker LLM across privacy representations.

Compares the deterministic low and high tiers with the base paper's
multi-agent rewrite (raw, and after the deterministic high-tier guard used by
run_pipeline.py). Run scripts/run_paper_agents.py first to include the rewrite.
Attacker answers are cached, so reruns cost no provider calls.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from adaptive_privacy.config import load_config
from adaptive_privacy.inference_attack import (
    GroqInferenceAttacker,
    evaluate_inference_attack,
    load_attack_cases,
    render_inference_report,
)
from adaptive_privacy.io import load_json, save_json
from adaptive_privacy.models import RiskLevel
from adaptive_privacy.tier_builder import TierBuilder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    results = PROJECT_ROOT / "results" / "adaptive_privacy"
    parser.add_argument("--rewrites", type=Path, default=results / "paper_agents" / "rewrites.json")
    parser.add_argument("--output-dir", type=Path, default=results / "inference_attack")
    parser.add_argument("--config", type=Path, default=None, help="adaptive_privacy JSON config")
    args = parser.parse_args()

    rewrites = load_json(args.rewrites) if args.rewrites.exists() else {}
    builder = TierBuilder()
    cases = []
    for case in load_attack_cases():
        variants = builder.build(case["document_id"], case["text"])
        versions = {"fixed_low": variants.low_text, "fixed_high": variants.high_text}
        rewrite = rewrites.get(case["document_id"], {}).get("rewritten_text")
        if rewrite:
            versions["paper_rewrite"] = rewrite
            versions["paper_rewrite_guarded"] = builder.sanitize(rewrite, RiskLevel.HIGH)
        cases.append({**case, "versions": versions})
    print(f"{len(cases)} records, {len(rewrites)} with paper rewrites")

    attacker = GroqInferenceAttacker(load_config(args.config), args.output_dir / "attack_cache.json")
    rows, summary = evaluate_inference_attack(cases, attacker)
    summary["benchmark"]["paper_rewrites"] = len(rewrites)
    print(f"{attacker.calls} provider call(s) made")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_json(args.output_dir / "summary.json", summary)
    with (args.output_dir / "evaluation.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "report.md").write_text(render_inference_report(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

"""Run the base paper's three-agent rewrite on the inference-attack records.

Rewrites are cached, so an interrupted or rate-limited run resumes where it
stopped. Uses configs/agents_config.yaml (Groq API mode by default).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from adaptive_privacy.inference_attack import load_attack_cases
from agents.system import PPRSystem, rewrite_cached


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "agents_config.yaml"))
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "adaptive_privacy" / "paper_agents" / "rewrites.json",
    )
    parser.add_argument("--chunk", type=int, default=3)
    args = parser.parse_args()

    cases = load_attack_cases()
    cache = rewrite_cached(
        PPRSystem(config_path=args.config),
        {case["document_id"]: case["text"] for case in cases},
        str(args.output),
        args.chunk,
    )
    missing = len(cases) - len(cache)
    if missing:
        print(f"{missing} record(s) failed; rerun this script to retry them.")


if __name__ == "__main__":
    main()

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
from adaptive_privacy.io import load_json, save_json
from agents.system import PPRSystem


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

    cache = load_json(args.output) if args.output.exists() else {}
    pending = [case for case in load_attack_cases() if case["document_id"] not in cache]
    print(f"{len(cache)} cached, {len(pending)} pending")
    system = PPRSystem(config_path=args.config)
    for start in range(0, len(pending), args.chunk):
        chunk = pending[start : start + args.chunk]
        results = system.process_documents_batch([case["text"] for case in chunk])
        for case, result in zip(chunk, results):
            # The agents return "" on provider or parsing errors; retry those next run.
            if result["rewritten_text"].strip():
                cache[case["document_id"]] = {
                    "rewritten_text": result["rewritten_text"],
                    "detected_pii": result["detected_pii"],
                }
        save_json(args.output, cache)
        print(f"{len(cache)} rewrites cached -> {args.output}")
    missing = len(load_attack_cases()) - len(cache)
    if missing:
        print(f"{missing} record(s) failed; rerun this script to retry them.")


if __name__ == "__main__":
    main()

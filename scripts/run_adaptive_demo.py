"""Run one query through the dependency-free adaptive privacy demo."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from adaptive_privacy.config import load_config
from adaptive_privacy.io import load_json
from adaptive_privacy.pipeline import AdaptivePrivacyPipeline
from adaptive_privacy.tier_builder import TierBuilder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "query",
        nargs="?",
        default="What treatment was recommended for the patient with diabetes?",
    )
    parser.add_argument("--role", default="patient")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument(
        "--mode",
        choices=["adaptive", "fixed_low", "fixed_high"],
        default="adaptive",
    )
    parser.add_argument(
        "--documents",
        type=Path,
        default=PROJECT_ROOT / "examples" / "adaptive_privacy" / "documents.json",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "adaptive_privacy.json",
    )
    parser.add_argument(
        "--audit",
        type=Path,
        default=PROJECT_ROOT / "results" / "adaptive_privacy" / "audit.jsonl",
    )
    args = parser.parse_args()

    documents = TierBuilder().build_many(load_json(args.documents))
    pipeline = AdaptivePrivacyPipeline(
        documents=documents,
        config=load_config(args.config),
        audit_path=str(args.audit),
    )
    result = pipeline.process(
        query=args.query,
        role=args.role,
        top_k=args.top_k,
        mode=args.mode,
    )
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

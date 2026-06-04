"""Build low, medium and high offline privacy variants from JSON documents."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from adaptive_privacy.io import load_json, save_tiered_documents
from adaptive_privacy.tier_builder import TierBuilder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "examples" / "adaptive_privacy" / "documents.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "adaptive_privacy" / "tiered_documents.json",
    )
    args = parser.parse_args()

    raw_documents = load_json(args.input)
    tiered_documents = TierBuilder().build_many(raw_documents)
    save_tiered_documents(args.output, tiered_documents)
    print(f"Built {len(tiered_documents)} tiered documents -> {args.output}")


if __name__ == "__main__":
    main()

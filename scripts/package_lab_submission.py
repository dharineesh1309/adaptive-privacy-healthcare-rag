"""Create a clean submission ZIP without raw data, models, caches, or audit logs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from adaptive_privacy.packaging import create_submission_archive


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "dist" / "adaptive_privacy_rag_lab.zip",
    )
    args = parser.parse_args()
    manifest = create_submission_archive(PROJECT_ROOT, args.output)
    print(
        json.dumps(
            {
                "archive": str(args.output),
                "files": manifest["files"],
                "bytes": args.output.stat().st_size,
                "raw_data_included": manifest["source_data_included"],
                "model_weights_included": manifest["model_weights_included"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

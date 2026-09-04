"""Launch the local adaptive privacy browser demo."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from adaptive_privacy.web_demo import create_application, create_server


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", choices=["synthetic", "chatdoctor"], default="synthetic"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Use only the local extractive answer generator.",
    )
    args = parser.parse_args()

    application = create_application(args.dataset, enable_llm=not args.no_llm)
    server = create_server(application, args.host, args.port)
    print(
        f"Adaptive privacy demo ({application.dataset_name}) -> "
        f"http://{args.host}:{args.port}"
    )
    generation = application.pipeline.generation_status()
    if generation.get("configured"):
        print(
            "Answer generation: "
            f"{generation.get('provider')} / {generation.get('model')} "
            f"({generation.get('calls_remaining', 'unlimited')} calls remaining today)"
        )
    else:
        print(
            "Answer generation: local extractive fallback "
            f"({generation.get('reason') or 'GROQ_API_KEY is not configured'})"
        )
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

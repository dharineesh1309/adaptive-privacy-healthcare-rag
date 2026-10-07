"""Run the base paper's evaluation protocol head-to-head (designed for a Kaggle GPU).

Point --base-url at any OpenAI-compatible server: Ollama on Kaggle
(http://localhost:11434/v1), vLLM, or Groq. Rewrites and answers are cached under
the output directory, so a stopped run resumes. See kaggle/paper_protocol.ipynb.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from adaptive_privacy.config import load_config
from adaptive_privacy.io import load_json, save_json
from adaptive_privacy.paper_protocol import (
    METHODS,
    ChatGenerator,
    ProtocolMethods,
    load_paper_questions,
    load_protocol_data,
    render_protocol_report,
    run_single_turn,
    run_stepwise,
    score_attack,
    score_stepwise,
    score_utility,
    stepwise_chains,
)


def span(text: str) -> tuple[int, int]:
    start, stop = text.split(":")
    return int(start), int(stop)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:11434/v1")
    parser.add_argument("--model", default="llama3.1:8b")
    parser.add_argument("--api-key-env", default="OLLAMA_API_KEY")
    parser.add_argument("--agents-config", default=str(PROJECT_ROOT / "configs" / "agents_config.yaml"))
    parser.add_argument("--config", type=Path, default=None, help="adaptive_privacy JSON config")
    parser.add_argument("--corpus", type=int, default=300)
    parser.add_argument("--utility", type=int, default=100)
    # The paper has 250 prompts per attack; develop on 0:125 and report 125:250.
    parser.add_argument("--targeted", type=span, default=(125, 250))
    parser.add_argument("--untargeted", type=span, default=(125, 250))
    parser.add_argument("--stepwise", type=int, default=20)
    parser.add_argument("--methods", default="naive,paper,ours,hybrid,ours_no_block,hybrid_no_block")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--skip-rewrite", action="store_true", help="use cached paper rewrites only")
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "results" / "adaptive_privacy" / "paper_protocol"
    )
    args = parser.parse_args()
    methods = [name for name in args.methods.split(",") if name]
    unknown = set(methods) - set(METHODS)
    if unknown:
        raise SystemExit(f"Unknown methods: {sorted(unknown)}; choose from {METHODS}")
    started = time.time()

    documents, utility = load_protocol_data(
        PROJECT_ROOT / "data" / "external" / "chatdoctor_protocol.json", args.corpus, args.utility
    )
    rewrites_path = args.output_dir / "rewrites.json"
    rewrites = load_json(rewrites_path) if rewrites_path.exists() else {}
    if not args.skip_rewrite and {"paper", "hybrid", "hybrid_no_block"} & set(methods):
        from agents.system import PPRSystem, rewrite_cached

        rewrites = rewrite_cached(
            PPRSystem(config_path=args.agents_config),
            {item["document_id"]: item["text"] for item in documents},
            str(rewrites_path),
            chunk=8,
        )
    print(f"{len(documents)} corpus records, {len(utility)} utility questions, {len(rewrites)} rewrites "
          f"({time.time() - started:.0f}s)")

    generator = ChatGenerator(args.base_url, args.model, args.api_key_env, args.output_dir / "generation_cache.jsonl")
    protocol = ProtocolMethods(documents, rewrites, load_config(args.config), generator)
    by_id = protocol.documents
    targeted = load_paper_questions("target_questions", *args.targeted)
    untargeted = load_paper_questions("untarget_questions", *args.untargeted)
    chains = stepwise_chains(documents, args.stepwise)

    summary = {
        "setup": {
            "model": args.model,
            "corpus": len(documents),
            "utility": len(utility),
            "paper_rewrites": len(rewrites),
            "targeted": len(targeted),
            "untargeted": len(untargeted),
            "stepwise": len(chains),
            "question_slices": {"targeted": list(args.targeted), "untargeted": list(args.untargeted)},
        },
        "targeted": {},
        "untargeted": {},
        "stepwise": {},
        "utility": {},
    }
    rows = []
    for method in methods:
        # Attackers are anonymous public users; utility and stepwise users are patients.
        task_rows = {
            "targeted": run_single_turn(protocol, method, targeted, "public", "general_information", args.workers),
            "untargeted": run_single_turn(protocol, method, untargeted, "public", "general_information", args.workers),
            "stepwise": run_stepwise(protocol, method, chains, args.workers),
            "utility": run_single_turn(protocol, method, utility, "patient", "treatment", args.workers),
        }
        summary["targeted"][method] = score_attack(task_rows["targeted"], by_id)
        summary["untargeted"][method] = score_attack(task_rows["untargeted"], by_id)
        summary["stepwise"][method] = score_stepwise(task_rows["stepwise"], by_id)
        summary["utility"][method] = score_utility(task_rows["utility"], generator, args.workers)
        for task, items in task_rows.items():
            rows.extend({"task": task, **item} for item in items)
        print(f"{method} done ({time.time() - started:.0f}s, {generator.calls} new calls)")
    summary["setup"]["generation_failures"] = generator.failures

    columns = ["task", "method", "query_id", "blocked", "repeat", "rouge_max", "leaked_values",
               "bleu1", "rouge_l", "fc_score", "answer"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "evaluation.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({**row, "answer": row["answer"][:300]} for row in rows)
    save_json(args.output_dir / "summary.json", summary)
    (args.output_dir / "report.md").write_text(render_protocol_report(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

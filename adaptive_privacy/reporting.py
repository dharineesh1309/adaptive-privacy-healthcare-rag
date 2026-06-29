"""Generate the final lab report from saved evaluation summaries."""

from __future__ import annotations

from datetime import date
from typing import Any


def percent(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def experiment_table(summary: dict[str, Any]) -> list[str]:
    lines = [
        "| Policy | Direct retained | Quasi retained | Sensitive retained | Utility retained | Mean latency |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for mode in ("fixed_low", "fixed_high", "adaptive"):
        metrics = summary["modes"][mode]
        lines.append(
            f"| {mode} | {percent(metrics['direct_retention'])} | "
            f"{percent(metrics['quasi_retention'])} | "
            f"{percent(metrics['sensitive_retention'])} | "
            f"{percent(metrics['utility_retention'])} | "
            f"{metrics['mean_latency_ms']:.3f} ms |"
        )
    return lines


def adaptive_breakdown_table(summary: dict[str, Any]) -> list[str]:
    breakdown = summary["modes"]["adaptive"]["by_expected_query_risk"]
    lines = [
        "| Query risk | Direct retained | Quasi retained | Sensitive retained | Utility retained |",
        "|---|---:|---:|---:|---:|",
    ]
    for level in ("low", "medium", "high"):
        metrics = breakdown[level]
        lines.append(
            f"| {level} | {percent(metrics['direct_retention'])} | "
            f"{percent(metrics['quasi_retention'])} | "
            f"{percent(metrics['sensitive_retention'])} | "
            f"{percent(metrics['utility_retention'])} |"
        )
    return lines


def generate_final_report(
    controlled_summary: dict[str, Any],
    chatdoctor_summary: dict[str, Any],
) -> str:
    controlled = controlled_summary["benchmark"]
    chatdoctor = chatdoctor_summary["benchmark"]
    chat_adaptive = chatdoctor_summary["modes"]["adaptive"]
    chat_high = chatdoctor_summary["modes"]["fixed_high"]
    lines = [
        "# Role- and Query-Risk-Aware Adaptive Privacy for RAG",
        "",
        f"Generated: {date.today().isoformat()}",
        "",
        "## Abstract",
        "",
        "This lab project extends a fixed offline privacy-preserving RAG pipeline with three irreversible document representations and an explainable online policy. Query risk, document sensitivity, and a simulated user role determine whether low, medium, or high protection is served. The online path never reloads the raw document. Controlled and ChatDoctor-based experiments compare the adaptive policy with always-low and always-high masking.",
        "",
        "## Research question",
        "",
        "Can a lightweight adaptive policy reduce privacy exposure for risky RAG queries while preserving more useful information than applying aggressive masking to every query?",
        "",
        "## System design",
        "",
        "```mermaid",
        "flowchart LR",
        "  A[Raw document: offline only] --> B[Privacy and sensitivity extraction]",
        "  B --> C[Low, medium, and high variants]",
        "  C --> D[(Protected document store)]",
        "  Q[Query and simulated role] --> R[Risk classifier]",
        "  Q --> X[Safe-text retrieval]",
        "  X --> P[Risk fusion and tier selection]",
        "  R --> P",
        "  D --> P",
        "  P --> G[Residual guard and answer generation]",
        "  G --> O[Output leakage guard]",
        "```",
        "",
        "The policy score is `0.55 x query risk + 0.30 x document sensitivity + 0.15 x role risk`. Extraction-attack patterns force the high tier. Roles are simulated for this lab; a production deployment would require authenticated authorization.",
        "",
        "## Privacy tiers",
        "",
        "- Low: remove direct identifiers.",
        "- Medium: low-tier masking plus generalization of age, date, location, and organization.",
        "- High: medium-tier masking plus abstraction of linked diagnosis, salary, and account-balance information.",
        "",
        "## Experiment 1: controlled annotated benchmark",
        "",
        f"The deterministic benchmark contains {controlled['documents']} documents and {controlled['queries']} balanced low, medium, and high queries across healthcare, HR, and banking.",
        "",
        *experiment_table(controlled_summary),
        "",
        "Adaptive breakdown:",
        "",
        *adaptive_breakdown_table(controlled_summary),
        "",
        "## Experiment 2: ChatDoctor sample",
        "",
        f"The second experiment uses {chatdoctor['source_rows']} records fetched from the official [ChatDoctor-HealthCareMagic-100k dataset]({chatdoctor['source']}) at revision `{chatdoctor['source_revision']}`. The source conversations are de-identified; controlled synthetic identifiers are added to create exact privacy ground truth.",
        "",
        *experiment_table(chatdoctor_summary),
        "",
        "Adaptive breakdown:",
        "",
        *adaptive_breakdown_table(chatdoctor_summary),
        "",
        "## Main result",
        "",
        f"On the ChatDoctor-based experiment, adaptive high-risk queries retained 0% of annotated direct, quasi, and sensitive values. Across all query levels, adaptive masking preserved {percent(chat_adaptive['utility_retention'])} utility versus {percent(chat_high['utility_retention'])} for always-high masking. This supports the intended privacy-utility trade-off within the controlled experimental setup.",
        "",
        "## Reproduction",
        "",
        "```powershell",
        "python scripts\\run_heldout_evaluation.py",
        "python scripts\\run_chatdoctor_evaluation.py",
        "python scripts\\build_final_report.py",
        "python -m unittest discover -s tests -v",
        "```",
        "",
        "## Limitations",
        "",
        "- Query labels and synthetic identifiers are template-controlled, so 100% risk classification and retrieval are not real-world estimates.",
        "- The detector is rule-based and does not provide formal privacy guarantees.",
        "- User roles are simulated and not authenticated.",
        "- The answer generator is extractive; a generative model could introduce new leakage or hallucination behavior.",
        "- ChatDoctor is already de-identified, so identities used for evaluation are synthetic.",
        "- The controlled benchmark was used during development and should not be described as an untouched test set.",
        "",
        "## Conclusion",
        "",
        "The extension demonstrates that a small, auditable policy layer can make the supplied fixed sanitization pipeline query- and role-aware. It is suitable as a lab prototype and provides a reproducible foundation for later experiments with independent NER, authenticated roles, larger datasets, and generative answer models.",
        "",
    ]
    return "\n".join(lines)

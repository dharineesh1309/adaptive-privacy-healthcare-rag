"""Generate the final lab report from saved evaluation summaries."""

from __future__ import annotations

from datetime import date
from typing import Any


def percent(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def experiment_table(summary: dict[str, Any]) -> list[str]:
    lines = [
        "| Policy | Direct retained | Quasi retained | Sensitive retained | Utility retained | Unique sensitive releases | Mean latency |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in ("fixed_low", "fixed_high", "adaptive_no_reid_check", "adaptive"):
        metrics = summary["modes"].get(mode)
        if metrics is None:
            continue
        lines.append(
            f"| {mode} | {percent(metrics['direct_retention'])} | "
            f"{percent(metrics['quasi_retention'])} | "
            f"{percent(metrics['sensitive_retention'])} | "
            f"{percent(metrics['utility_retention'])} | "
            f"{percent(metrics.get('unique_sensitive_release_rate', 0.0))} | "
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
    session_summary: dict[str, Any] | None = None,
    challenge_summary: dict[str, Any] | None = None,
    external_injection_summary: dict[str, Any] | None = None,
    ledger_summary: dict[str, Any] | None = None,
    inference_summary: dict[str, Any] | None = None,
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
        "This lab project extends a fixed offline privacy-preserving RAG pipeline with three irreversible document representations, verified-role gating, purpose-bound disclosure rules, retrieved-context prompt-injection neutralization, cumulative session risk, and a persistent pseudonymous privacy ledger. Query risk, document sensitivity, role assurance, declared purpose, and prior cross-session exposure determine whether low, medium, or high protection is served or the request is blocked. The online path never reloads the raw document. Controlled and ChatDoctor-based experiments compare adaptive masking with always-low and always-high policies, while adversarial conversations and ablations test the defense-in-depth layer. A release-time k-anonymity check raises the tier when a sensitive value would appear in a record that is unique on its visible quasi-identifiers, and an attacker-LLM experiment measures whether masked diagnoses can still be inferred, including from the base paper's own multi-agent rewrite.",
        "",
        "## Research question",
        "",
        "Can a lightweight adaptive policy reduce privacy exposure for risky RAG queries, preserve more useful information than always-high masking, and prevent accumulated disclosure from being reset by opening a new session?",
        "",
        "## System design",
        "",
        "```mermaid",
        "flowchart LR",
        "  RAW[Raw document: offline only] --> B[Privacy and sensitivity extraction]",
        "  B --> C[Low, medium, and high variants]",
        "  C --> D[(Protected document store)]",
        "  Q[Query, role, purpose, principal] --> A[Role assurance]",
        "  A --> R[Risk and injection classifier]",
        "  Q --> X[Safe-text retrieval]",
        "  R --> C[Purpose and attribute-flow policy]",
        "  X --> I[Retrieved-content integrity guard]",
        "  I --> S[Session risk budget]",
        "  C --> L[Persistent pseudonymous ledger]",
        "  S --> L",
        "  L -->|allow or escalate| P[Risk fusion and tier selection]",
        "  L -->|block| Z[Safe refusal]",
        "  R --> P",
        "  D --> P",
        "  P --> G[Residual guard and answer generation]",
        "  G --> O[Output leakage guard]",
        "```",
        "",
        "The tier score is `0.55 x query risk + 0.30 x document sensitivity + 0.15 x role risk`. Extraction and prompt-injection patterns force the high tier. Unverified doctor/admin claims are downgraded to public policy, purpose rules restrict role-attribute flows, and instruction-like segments found in retrieved text are removed before generation. The persistent ledger records only hashed principal/document references and attribute categories; it does not store query text or identifier values. Role verification remains simulated for this lab; production deployment requires authenticated authorization middleware.",
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
    ]
    chat_no_reid = chatdoctor_summary["modes"].get("adaptive_no_reid_check")
    if chat_no_reid is not None:
        lines.extend(
            [
                f"Release-time re-identification check: without it, {percent(chat_no_reid['unique_sensitive_release_rate'])} of adaptive ChatDoctor releases showed a sensitive value in a record unique on its visible quasi-identifiers (age range and year are enough to single out a record). With the k>=2 check this fell to {percent(chat_adaptive['unique_sensitive_release_rate'])}, while utility moved from {percent(chat_no_reid['utility_retention'])} to {percent(chat_adaptive['utility_retention'])}. Verified clinicians are exempt because identifying their patient is the purpose of their access.",
                "",
            ]
        )
    if session_summary is not None:
        session_metrics = session_summary["metrics"]
        session_benchmark = session_summary["benchmark"]
        lines.extend(
            [
                "## Experiment 3: cumulative multi-turn privacy",
                "",
                f"The deterministic session benchmark contains {session_benchmark['scenarios']} conversations: {session_benchmark['attack_scenarios']} slow-extraction or adversarial scenarios and {session_benchmark['benign_scenarios']} benign scenarios. It covers medical and financial attribute aggregation, repetition, direct prompt injection, transformation-based evasion, role impersonation, and role switching. The guard uses decayed cumulative risk, repeated document access, requested-attribute diversity, role assurance, and prior blocked attempts. It stores no raw query text or identifier values in session state.",
                "",
                "| Metric | Result |",
                "|---|---:|",
                f"| Attack block rate | {percent(session_metrics['attack_block_rate'])} |",
                f"| Benign block rate | {percent(session_metrics['benign_block_rate'])} |",
                f"| Mean turn to escalation | {session_metrics['mean_turn_to_escalation']:.2f} |",
                f"| Mean turn to block | {session_metrics['mean_turn_to_block']:.2f} |",
                "",
                f"All scripted attack scenarios were blocked in this benchmark ({percent(session_metrics['attack_block_rate'])}), while none of the benign conversations was blocked ({percent(session_metrics['benign_block_rate'])} benign block rate). These are deterministic functional scenarios, not a security guarantee or an estimate of real-world attack prevalence.",
                "",
            ]
        )
        if session_summary.get("ablations"):
            lines.extend(
                [
                    "### Signal ablation",
                    "",
                    "| Variant | Attack block rate | Benign block rate | Mean turn to block |",
                    "|---|---:|---:|---:|",
                ]
            )
            for item in session_summary["ablations"]:
                mean_block = item["mean_turn_to_block"]
                mean_block_text = f"{mean_block:.2f}" if mean_block is not None else "-"
                lines.append(
                    f"| {item['variant']} | {percent(item['attack_block_rate'])} | "
                    f"{percent(item['benign_block_rate'])} | {mean_block_text} |"
                )
            rates = {
                item["variant"]: percent(item["attack_block_rate"])
                for item in session_summary["ablations"]
            }
            lines.extend(
                [
                    "",
                    f"Removing behavioral memory changed the attack block rate from {rates.get('full_guard', '-')} to {rates.get('no_behavioral_memory', '-')} in this scripted benchmark, and removing the explicit attack-pattern signal gave {rates.get('no_attack_pattern_signal', '-')}. Attacks still blocked without memory are stopped by the per-request purpose policy, so cumulative session state and purpose binding contribute separately.",
                    "",
                ]
            )
        lines.extend(
            [
                "### Research basis",
                "",
                "The design follows the same broad principle as selective-disclosure RAG research: enforce disclosure before generation instead of trusting an LLM instruction. Multi-turn prompt-leakage studies motivate cumulative evaluation, while NIST guidance motivates deterministic authorization and monitoring controls. See the linked research matrix in `docs/research_scope.md` for scope boundaries and sources.",
                "",
            ]
        )
    if challenge_summary is not None:
        challenge_benchmark = challenge_summary["benchmark"]
        challenge_metrics = challenge_summary["overall"]
        baseline_metrics = challenge_summary["baseline_phrase_only"]["overall"]
        lines.extend(
            [
                "## Experiment 4: frozen multilingual injection challenge",
                "",
                f"The versioned challenge contains {challenge_metrics['cases']} balanced attack/benign queries in English, Hindi, and Kannada. Its fixture SHA-256 is `{challenge_benchmark['sha256']}`. Full challenge sentences are absent from the configured exact-phrase list. The comparison evaluates the original phrase-only detector and the compositional detector against the same frozen artifact.",
                "",
                "| Detector | Attack recall | Attack precision | Benign false-positive rate | Accuracy |",
                "|---|---:|---:|---:|---:|",
                f"| Exact-phrase baseline | {percent(baseline_metrics['attack_recall'])} | {percent(baseline_metrics['attack_precision'])} | {percent(baseline_metrics['benign_false_positive_rate'])} | {percent(baseline_metrics['accuracy'])} |",
                f"| Compositional multilingual guard | {percent(challenge_metrics['attack_recall'])} | {percent(challenge_metrics['attack_precision'])} | {percent(challenge_metrics['benign_false_positive_rate'])} | {percent(challenge_metrics['accuracy'])} |",
                "",
                "The phrase baseline missed every paraphrased attack. The compositional guard combines action and target concepts and detected all attacks in this small curated challenge without flagging its educational controls. This is a regression result, not evidence of general multilingual robustness.",
                "",
            ]
        )
    if external_injection_summary is not None:
        external_benchmark = external_injection_summary["benchmark"]
        external_phrase = external_injection_summary["phrase_baseline"]["overall"]
        external_rules = external_injection_summary["compositional_guard"]["overall"]
        external_learned = external_injection_summary["learned_tfidf_centroid"]["overall"]
        learned_model = external_injection_summary["learned_model"]
        lines.extend(
            [
                "## Experiment 5: external scenario-isolated injection test",
                "",
                f"The external evaluation uses the {external_benchmark['rows']}-row test split of [{external_benchmark['dataset_id']}]({external_benchmark['dataset_page']}) at revision `{external_benchmark['revision']}` under {external_benchmark['license']}. Its published train, validation, and test templates are scenario-isolated. Raw rows are cached locally and excluded from the submission archive.",
                "",
                "| Detector | Recall | Precision | Benign false-positive rate | Accuracy |",
                "|---|---:|---:|---:|---:|",
                f"| Exact-phrase baseline | {percent(external_phrase['attack_recall'])} | {percent(external_phrase['attack_precision'])} | {percent(external_phrase['benign_false_positive_rate'])} | {percent(external_phrase['accuracy'])} |",
                f"| Compositional guard | {percent(external_rules['attack_recall'])} | {percent(external_rules['attack_precision'])} | {percent(external_rules['benign_false_positive_rate'])} | {percent(external_rules['accuracy'])} |",
                f"| Learned char TF-IDF centroid | {percent(external_learned['attack_recall'])} | {percent(external_learned['attack_precision'])} | {percent(external_learned['benign_false_positive_rate'])} | {percent(external_learned['accuracy'])} |",
                "",
                f"The learned detector used {learned_model['training_rows']} training rows and {learned_model['validation_rows']} validation rows; test labels were not used for fitting or threshold selection. Although validation false-positive rate was {percent(learned_model['validation_metrics']['false_positive_rate'])}, test false-positive rate rose to {percent(external_learned['benign_false_positive_rate'])}. It is therefore reported as a failed generalization baseline and is not deployed in the live pipeline.",
                "",
                "The external test materially changes the conclusion: curated regression success does not imply broad agentic-injection coverage. The conservative compositional guard remains useful as one defense layer, but neither rule matching nor this small learned baseline is sufficient alone.",
                "",
            ]
        )
    if ledger_summary is not None:
        ledger_benchmark = ledger_summary["benchmark"]
        ledger_metrics = ledger_summary["metrics"]
        lines.extend(
            [
                "## Experiment 6: purpose-bound persistent privacy ledger",
                "",
                f"The controlled ledger benchmark contains {ledger_benchmark['scenarios']} scenarios covering a cross-session reset attack, duplicate disclosure, principal isolation, a role-purpose violation, and benign general information. It compares the full guard with a session-only ablation whose persistent budgets are effectively disabled.",
                "",
                "| Metric | Result |",
                "|---|---:|",
                f"| Full scenario pass rate | {percent(ledger_metrics['full_guard_pass_rate'])} |",
                f"| Persistent-ledger reset-attack block rate | {percent(ledger_metrics['persistent_reset_attack_block_rate'])} |",
                f"| Session-only reset-attack block rate | {percent(ledger_metrics['session_only_reset_attack_block_rate'])} |",
                f"| Duplicate exposure charge | {ledger_metrics['duplicate_exposure_charge']} |",
                f"| Raw values absent from saved ledger | {'yes' if ledger_metrics['raw_values_absent_from_ledger'] else 'no'} |",
                "",
                "The same principal could not evade its disclosure budget by starting a fresh session, while a different principal retained an independent budget and replaying an already-accounted disclosure incurred no additional charge. The ablation allowed the reset attack, isolating the contribution of cross-session persistence in this deterministic setup.",
                "",
                "### Research basis",
                "",
                "The purpose policy operationalizes contextual integrity as a role-purpose-attribute flow check. The ledger applies a multi-query accounting idea at an application-policy level: marginal attribute exposures are accumulated before generation across otherwise independent sessions. It is not differential privacy and makes no formal privacy claim.",
                "",
            ]
        )
    if inference_summary is not None:
        benchmark = inference_summary["benchmark"]
        lines.extend(
            [
                "## Experiment 7: attacker-LLM diagnosis inference",
                "",
                f"Literal-retention metrics only check whether an annotated string survives masking. Here an LLM adversary reads each released representation of {benchmark['records']} medical records ({', '.join(benchmark['datasets'])}) and guesses the hidden diagnosis from remaining clues. The base paper's three-agent rewrite was produced through the same provider for {benchmark.get('paper_rewrites', 0)} records; `paper_rewrite_guarded` is that rewrite after the deterministic high-tier scan, as `run_pipeline.py` uses it.",
                "",
                "| Representation | Diagnosis string visible | Attacker recovered diagnosis |",
                "|---|---:|---:|",
            ]
        )
        for version, item in inference_summary["versions"].items():
            lines.append(
                f"| {version} | {percent(item['literal_visible_rate'])} | {percent(item['recovery_rate'])} |"
            )
        lines.extend(
            [
                "",
                f"Without any record the attacker guessed `{inference_summary['prior']['guess']}`, matching {percent(inference_summary['prior']['recovery_rate'])} of records by chance. The gap between the two columns is inferential leakage through medications, symptoms, and therapies, which literal retention does not measure. Scoring is a conservative lexical match, so recovery rates are lower bounds.",
                "",
            ]
        )
    lines.extend(
        [
        "## Reproduction",
        "",
        "```powershell",
        "python scripts\\run_heldout_evaluation.py",
        "python scripts\\run_chatdoctor_evaluation.py",
        "python scripts\\evaluate_session_privacy.py",
        "python scripts\\evaluate_frozen_challenge.py",
        "python scripts\\evaluate_external_injection.py",
        "python scripts\\evaluate_privacy_ledger.py",
        "# Needs GROQ_API_KEY; answers are cached under results\\adaptive_privacy",
        "python scripts\\run_paper_agents.py",
        "python scripts\\evaluate_inference_attack.py",
        "python scripts\\build_final_report.py",
        "python -m unittest discover -s tests -v",
        "```",
        "",
        "## Limitations",
        "",
        "- Query labels and synthetic identifiers are template-controlled, so 100% risk classification and retrieval are not real-world estimates.",
        "- The detector is rule-based and does not provide formal privacy guarantees.",
        "- Role assurance is simulated with a boolean flag and is not authentication.",
        "- The answer generator is extractive; a generative model could introduce new leakage or hallucination behavior.",
        "- ChatDoctor is already de-identified, so identities used for evaluation are synthetic.",
        "- The controlled benchmark was used during development and should not be described as an untouched test set.",
        "- The session benchmark contains ten scripted conversations and does not prove resistance to adaptive attackers.",
        "- Prompt-injection detection uses compositional lexicons and will miss novel semantic, obfuscated, or unsupported-language attacks.",
        "- The multilingual challenge is small, curated during development, and covers only English, Hindi, and Kannada.",
        "- The external benchmark is synthetic, English-only, and its scenario-isolated test distribution caused substantial learned-model overblocking.",
        "- The purpose and ledger benchmark has five controlled scenarios; budgets and purpose rules are hand-configured rather than learned or legally validated.",
        "- The persistent ledger is application-level disclosure accounting, not differential privacy, and principal IDs require trustworthy identity binding in production.",
        "- The ledger-reset endpoint is a lab control; it must be removed or restricted to authorized administrators in production.",
        "- Fixed-low/fixed-high modes and the synthetic comparison endpoint are evaluation controls and must not be exposed as user-selectable production policy overrides.",
        "- k-anonymity is computed over the loaded corpus with detector-found quasi-identifiers; in small corpora most records are unique, so the check escalates often.",
        "- The inference attack uses one attacker model and lexical scoring on 22 records; a stronger attacker or semantic judging would find more leakage, not less.",
        "",
        "## Conclusion",
        "",
        "The extension demonstrates that a small, auditable defense-in-depth layer can make the supplied fixed sanitization pipeline query-, role-, purpose-, session-, cross-session-, and prompt-injection-aware. The reset-attack ablation isolates why persistent disclosure accounting matters, while the external injection result keeps the security claim appropriately narrow. It is suitable as a lab prototype and provides a reproducible foundation for authenticated identity, independently validated purpose rules, stronger detectors, and generative-model red teaming.",
        "",
        ]
    )
    return "\n".join(lines)

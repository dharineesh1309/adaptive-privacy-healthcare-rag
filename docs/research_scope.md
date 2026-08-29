# Research Scope and Extension Rationale

## What the upstream project contributes

The upstream [Privacy-Preserving-RAG repository](https://github.com/foursoils/Privacy-Preserving-RAG)
uses an offline multi-agent rewriting pipeline. Its central strength is that privacy
transformation occurs before online generation. Its main limitation for this lab is
that every query receives the same stored rewrite and the online path has no
session-level disclosure budget or authorization assurance.

## Evidence used to choose the extension

| Source | Relevant finding | Implemented response |
|---|---|---|
| [SD-RAG](https://arxiv.org/abs/2601.11199) | Selective disclosure should be enforced before generation instead of relying only on LLM instructions. | Deterministic tier selection and retrieved-context neutralization occur before generation. |
| [Prompt Leakage in Multi-turn LLM Applications](https://aclanthology.org/2024.emnlp-industry.94/) | Multi-turn interaction can substantially increase prompt-leakage attack success. | Cumulative session score, repeated-access signal, privacy budgets, and multi-turn benchmark. |
| [Your LLM Agent Can Leak Your Data](https://aclanthology.org/2026.findings-acl.1257/) | Multi-turn interaction can amplify sustained and cumulative exfiltration. | Slow attribute-aggregation scenarios across medical, HR/role, and financial behavior. |
| [NIST AI 600-1](https://doi.org/10.6028/NIST.AI.600-1) | GenAI risk management should address privacy, prompt injection, monitoring, and system-level safeguards. | Privacy-safe audit metadata, deterministic guards, and explicit limitations. |
| [NIST Agent Identity design tenets](https://www.nist.gov/blogs/cybersecurity-insights/design-tenets-agent-identity-management-systems) | Authorization should use deterministic decisions and continuous runtime verification. | Privileged role claims default to public until verified; session policy is re-evaluated every turn. |
| [RAG-PIBench](https://arxiv.org/abs/2610.08571) | Prompt-injection detection needs leakage-aware benchmarks and strong baselines. | Phrase-based baseline is measured through injection/evasion scenarios and documented as a baseline, not a solved detector. |
| [Agentic Prompt-Injection 5K](https://huggingface.co/datasets/3nesdeniz/agentic-prompt-injection-5k) | Paired attacks and benign hard negatives are separated by scenario templates across train, validation, and test. | The external test split evaluates rules and a learned character TF-IDF centroid without using test labels for fitting or threshold selection. |
| [Need to Know: Contextual-Integrity-Grounded Query Rewriting](https://aclanthology.org/2026.emnlp-main.300/) | Type-only PII masking is context-agnostic; privacy decisions should retain only information necessary for the task. | A deterministic role-purpose-attribute policy blocks information flows that are unnecessary or unauthorized for the declared task. |
| [From One Query to Many: Privacy Accounting in RAG](https://arxiv.org/abs/2610.21245) | Repeated-query privacy claims require an explicit protected unit, accounting scope, and release boundary. | The protected unit is an attribute category per document, the scope is a pseudonymous principal across sessions, and the release boundary is the context sent to generation. This is application-level accounting, not DP. |

## Implemented scope

1. Query-, role-, document-, and session-aware privacy-tier selection.
2. Cumulative privacy budget with decay, repetition, attribute-diversity, role-change,
   role-claim, and attack-pattern signals.
3. Conservative privileged-role assurance.
4. Direct query prompt-injection detection.
5. Retrieved-context instruction neutralization before generation.
6. Ten deterministic conversations spanning three benign flows and seven attack
   flows.
7. Five ablation variants that isolate behavioral memory, attack-pattern,
   repetition, and role signals.
8. A versioned, SHA-256-fingerprinted challenge of 30 balanced English, Hindi,
   and Kannada paraphrases. It compares the original exact-phrase baseline with a
   compositional action-plus-target detector.
9. An external 545-row scenario-isolated test using a fixed dataset revision and
   CC-BY-4.0 attribution. Raw source rows remain outside the submission archive.
10. A dependency-free character TF-IDF centroid baseline trained on 3,923 rows and
    thresholded on 467 validation rows. Its high external benign false-positive
    rate is retained as a negative generalization result rather than hidden.
11. Contextual-integrity policy over effective role, declared purpose, and
    requested attribute types.
12. A persistent disclosure ledger keyed by a SHA-256 principal reference, with
    hashed document references and no raw queries or private values.
13. A five-scenario cross-session benchmark and session-only ablation covering
    reset evasion, idempotent replay, principal isolation, purpose denial, and
    benign access.

## Best next scopes, in priority order

1. **Authenticated principal binding.** Replace the user-entered lab principal and
   role-verification boolean with signed identity claims, authorization checks,
   ledger retention policy, and administrative recovery controls.
2. **External protected test split.** The bundled frozen challenge is still curated
   during development. A test set authored by another person, or a public protected
   split, would provide stronger scientific credibility.
3. **Independently validated purpose rules.** Evaluate the role-purpose-attribute
   matrix with domain experts and measure both inappropriate disclosure and denied
   legitimate access.
4. **Better calibrated learned detector.** Compare class-weighted logistic
   regression or a compact transformer against the centroid baseline, with explicit
   false-positive constraints and an additional out-of-distribution validation set.
5. **Generative-model red teaming.** Replace the extractive generator with a local
   or API LLM and measure both verbatim and semantic leakage.
6. **Formal privacy mechanisms.** Differential privacy or private information
   retrieval would provide stronger guarantees, but they are beyond a three-day lab
   implementation and should not be claimed by this prototype.

## Claim boundary

The project demonstrates deterministic functional behavior on controlled data. It
does not prove privacy, authenticate real users, cover arbitrary semantic or
multilingual injections, or estimate real-world attack prevalence. The frozen
challenge is a regression set, not an independent external benchmark. The ablation
only shows that the scripted result depends on cumulative signals rather than on
the single-turn classifier alone. The persistent ledger is deterministic policy
accounting and must not be described as a formal differential-privacy guarantee.

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

## Best next scopes, in priority order

1. **Independent protected test split.** Freeze unseen paraphrases and multilingual
   attacks before tuning any thresholds. This is the best next improvement for
   scientific credibility.
2. **Real identity middleware.** Replace the lab boolean with signed role claims and
   document-level access-control lists.
3. **Learned injection detector.** Compare the present phrase baseline with TF-IDF
   logistic regression or a compact transformer on a public benchmark.
4. **Generative-model red teaming.** Replace the extractive generator with a local
   or API LLM and measure both verbatim and semantic leakage.
5. **Cross-session pseudonymous ledger.** Bind privacy budgets to a privacy-preserving
   principal identifier so opening a new browser session cannot reset risk.
6. **Formal privacy mechanisms.** Differential privacy or private information
   retrieval would provide stronger guarantees, but they are beyond a three-day lab
   implementation and should not be claimed by this prototype.

## Claim boundary

The project demonstrates deterministic functional behavior on controlled data. It
does not prove privacy, authenticate real users, cover novel semantic injections,
or estimate real-world attack prevalence. The ablation only shows that the scripted
result depends on the cumulative signals rather than on the single-turn classifier
alone.

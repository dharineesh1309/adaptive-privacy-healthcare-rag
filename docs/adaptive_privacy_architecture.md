# Adaptive Privacy Architecture

## Objective

Extend the original one-size-fits-all sanitization pipeline with an explainable
runtime policy while ensuring that raw documents never enter the online query
path.

## Trust boundary

- Raw documents are available only to offline preparation.
- Offline preparation emits low, medium, and high privacy variants.
- The online index uses the low-tier text, in which direct identifiers are already
  masked.
- Online policy selection can only choose the same or a more private stored
  representation. It cannot reconstruct removed values.
- Audit logs contain a query hash and policy metadata, not the query, contexts, or
  answer.
- Doctor/admin claims are treated as public until a trusted external layer marks
  them verified. The bundled web checkbox only simulates that upstream decision.
- Retrieved text is data, not authority: instruction-like segments are removed
  before the selected context reaches generation.
- Purpose is evaluated against the effective role and requested attribute types;
  declaring a purpose does not grant a role new authority.
- Cross-session state contains only SHA-256 principal/document references,
  attribute categories, purpose labels, counters, and timestamps. It does not
  store raw principal IDs, query text, contexts, or private values.
- Fixed-low/fixed-high and the raw three-way comparison are synthetic evaluation
  controls. The normal query endpoint always enforces purpose and ledger policy,
  even when multi-turn session memory is disabled.
- The ledger-reset endpoint exists only to make experiments reproducible. A real
  deployment must remove it or authorize it as a privileged administrative action.

## Offline flow

1. Detect direct identifiers, quasi-identifiers, and sensitive attributes.
2. Calculate a document-sensitivity score.
3. Produce three irreversible variants:
   - Low: direct identifiers are masked.
   - Medium: direct identifiers are masked and quasi-identifiers are generalized.
   - High: medium protection plus sensitive-attribute abstraction.
4. Store variants under a stable document ID.
5. Build the retrieval index from low-tier text.

## Online flow

1. Resolve the requested role. Unverified privileged roles are downgraded to
   public policy.
2. Classify query and prompt-injection risk using explainable patterns.
3. Infer requested attribute types and check the declared purpose against the
   effective role and configured information-flow rule.
4. Assign an effective-role risk value. Unknown roles are treated as public.
5. Remove attack/control phrases from the retrieval query so they do not distort
   relevance.
6. Retrieve document IDs from the low-tier index.
7. Neutralize prompt-like instructions in retrieved text.
8. Update cumulative session risk and privacy budgets from behavioral signals.
9. Preview the marginal quasi/sensitive attribute exposure against the
   pseudonymous principal's persistent budget. Starting a new session does not
   reset this budget; a repeated identical disclosure is not charged twice.
10. Fuse query, document, and role risk:

   `risk = 0.55 * query + 0.30 * document + 0.15 * role`

11. Select the low, medium, or high stored variant. Extraction-attack patterns
   always force high.
12. Run a residual privacy scan before generation.
13. Run another privacy scan over the generated answer.
14. On an allowed response, record value-free exposure categories in the ledger
   and non-sensitive decision metadata in the audit log.

## Failure defaults

- Unknown role: public risk.
- Unverified doctor/admin role: public risk plus a session-risk signal.
- Unknown purpose or invalid principal identifier: reject the request.
- Disallowed role-purpose-attribute flow: block before generation.
- Exhausted persistent principal budget: escalate the tier or block a targeted
  disclosure before generation.
- Direct prompt-injection pattern: high tier.
- Retrieved prompt-injection pattern: remove the instruction-like segment.
- Extraction-pattern match: high tier.
- Missing relevant context: return no-context response.
- Three or more residual direct identifiers in generated text: block the answer.
- Empty or invalid source document: reject it during tier construction.

## Evaluation

The experiment compares fixed-low, fixed-high, and adaptive policies using:

- direct-identifier retention;
- all-sensitive-value retention;
- non-direct token utility retention;
- query-risk classification accuracy;
- selected-tier distribution;
- added latency;
- multi-turn attack and benign block rates;
- signal-ablation changes in attack block rate;
- frozen multilingual injection recall, precision, and benign false-positive rate;
- external scenario-isolated injection recall and false-positive rate;
- train/validation/test generalization of a learned character TF-IDF baseline;
- cross-session reset-attack blocking with a session-only ablation;
- duplicate-disclosure charging, purpose denial, and principal isolation; and
- absence of raw values from the saved pseudonymous ledger.

The bundled synthetic benchmark is intended for functionality and regression
testing. Independent NER, human review, and a larger held-out dataset are required
before making empirical privacy claims.

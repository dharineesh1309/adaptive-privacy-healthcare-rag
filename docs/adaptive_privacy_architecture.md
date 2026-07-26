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
3. Assign an effective-role risk value. Unknown roles are treated as public.
4. Remove attack/control phrases from the retrieval query so they do not distort
   relevance.
5. Retrieve document IDs from the low-tier index.
6. Neutralize prompt-like instructions in retrieved text.
7. Update cumulative session risk and privacy budgets from behavioral signals.
8. Fuse query, document, and role risk:

   `risk = 0.55 * query + 0.30 * document + 0.15 * role`

9. Select the low, medium, or high stored variant. Extraction-attack patterns
   always force high.
10. Run a residual privacy scan before generation.
11. Run another privacy scan over the generated answer.
12. Record non-sensitive decision metadata in the audit log.

## Failure defaults

- Unknown role: public risk.
- Unverified doctor/admin role: public risk plus a session-risk signal.
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
- multi-turn attack and benign block rates; and
- signal-ablation changes in attack block rate.

The bundled synthetic benchmark is intended for functionality and regression
testing. Independent NER, human review, and a larger held-out dataset are required
before making empirical privacy claims.

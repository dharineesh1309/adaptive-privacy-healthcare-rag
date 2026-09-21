"""Release-time k-anonymity: is this record unique on what the tier leaves visible?

For a document released at a tier, k counts corpus records whose visible
quasi-identifiers at that tier include the same values. An attacker who knows a
target's age range and year (for example) cannot single out the target when
k >= 2. A sensitive attribute shown in a k=1 release is linkable to one person.
"""

from __future__ import annotations

import re

from .models import DocumentVariants, EntitySpan, RiskLevel
from .tier_builder import TierBuilder


class ReidentificationRisk:
    # ponytail: O(n^2) precompute over the corpus; index signatures by value for large stores.
    def __init__(self, documents: list[DocumentVariants]):
        self.signatures: dict[RiskLevel, dict[str, frozenset]] = {
            tier: {item.document_id: self._signature(item, tier) for item in documents}
            for tier in RiskLevel
        }
        self.sensitive = {
            item.document_id: any(
                entity.get("category") == "sensitive" for entity in item.detected_entities
            )
            for item in documents
        }
        self.k_values = {
            tier: {
                document_id: sum(other >= signature for other in signatures.values())
                for document_id, signature in signatures.items()
            }
            for tier, signatures in self.signatures.items()
        }

    def k(self, document_id: str, tier: RiskLevel) -> int | None:
        return self.k_values[tier].get(document_id)

    def exposes_sensitive(self, document_id: str, tier: RiskLevel) -> bool:
        return tier is not RiskLevel.HIGH and self.sensitive.get(document_id, False)

    @staticmethod
    def _signature(document: DocumentVariants, tier: RiskLevel) -> frozenset:
        values = set()
        for entity in document.detected_entities:
            if entity.get("category") != "quasi":
                continue
            if tier is RiskLevel.LOW:
                value = entity["value"]
                if entity["entity_type"] == "age":
                    value = re.sub(r"\D", "", value)
            else:
                value = TierBuilder._replacement(EntitySpan(**entity), tier)
                # [REGION], [ORGANIZATION], [DATE] carry no value; ranges/years do.
                if not re.search(r"\d", value):
                    continue
            values.add((entity["entity_type"], value.casefold()))
        return frozenset(values)

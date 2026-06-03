"""Create irreversible low, medium and high privacy document variants."""

from __future__ import annotations

import re
from typing import Iterable

from .detector import PrivacyDetector
from .models import DocumentVariants, EntitySpan, RiskLevel


DIRECT_REPLACEMENTS = {
    "name": "[PERSON]",
    "email": "[EMAIL]",
    "phone": "[PHONE]",
    "address": "[ADDRESS]",
    "government_id": "[IDENTIFIER]",
    "account_number": "[ACCOUNT]",
}

QUASI_REPLACEMENTS = {
    "location": "[REGION]",
    "organization": "[ORGANIZATION]",
}

SENSITIVE_REPLACEMENTS = {
    "diagnosis": "[MEDICAL CONDITION]",
    "salary": "[FINANCIAL INFORMATION]",
}


class TierBuilder:
    def __init__(self, detector: PrivacyDetector | None = None):
        self.detector = detector or PrivacyDetector()

    def build(
        self,
        document_id: str,
        text: str,
        metadata: dict | None = None,
    ) -> DocumentVariants:
        entities = self.detector.detect(text)
        return DocumentVariants(
            document_id=str(document_id),
            sensitivity_score=self.detector.sensitivity_score(entities),
            low_text=self.sanitize(text, RiskLevel.LOW, entities),
            medium_text=self.sanitize(text, RiskLevel.MEDIUM, entities),
            high_text=self.sanitize(text, RiskLevel.HIGH, entities),
            detected_entities=[entity.to_dict() for entity in entities],
            metadata=dict(metadata or {}),
        )

    def build_many(self, documents: Iterable[dict]) -> list[DocumentVariants]:
        variants = []
        for document in documents:
            document_id = document.get("document_id", document.get("id"))
            if document_id is None:
                raise ValueError("Every document needs 'document_id' or 'id'.")
            text = document.get("text", document.get("content"))
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Document {document_id!r} has no non-empty text.")
            variants.append(
                self.build(document_id, text, document.get("metadata", {}))
            )
        return variants

    def sanitize(
        self,
        text: str,
        level: RiskLevel | str,
        entities: list[EntitySpan] | None = None,
    ) -> str:
        selected_level = RiskLevel.from_value(level)
        detected = entities if entities is not None else self.detector.detect(text)
        allowed_categories = {"direct"}
        if selected_level in (RiskLevel.MEDIUM, RiskLevel.HIGH):
            allowed_categories.add("quasi")
        if selected_level is RiskLevel.HIGH:
            allowed_categories.add("sensitive")

        output = text
        for entity in reversed(detected):
            if entity.category not in allowed_categories:
                continue
            replacement = self._replacement(entity, selected_level)
            output = output[: entity.start] + replacement + output[entity.end :]
        return self._normalize(output)

    @staticmethod
    def _replacement(entity: EntitySpan, level: RiskLevel) -> str:
        if entity.category == "direct":
            return DIRECT_REPLACEMENTS.get(entity.entity_type, "[PRIVATE]")
        if entity.entity_type == "age":
            number_match = re.search(r"\d{1,3}", entity.value)
            if not number_match:
                return "[AGE RANGE]"
            age = int(number_match.group())
            lower = (age // 5) * 5
            return f"[AGE {lower}-{lower + 4}]"
        if entity.entity_type == "date":
            year_match = re.search(r"\b(?:19|20)\d{2}\b", entity.value)
            return f"[YEAR {year_match.group()}]" if year_match else "[DATE]"
        if entity.category == "quasi":
            return QUASI_REPLACEMENTS.get(entity.entity_type, "[GENERALIZED]")
        if level is RiskLevel.HIGH:
            return SENSITIVE_REPLACEMENTS.get(entity.entity_type, "[SENSITIVE]")
        return entity.value

    @staticmethod
    def _normalize(text: str) -> str:
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\s+([,.;:])", r"\1", text)
        return text.strip()

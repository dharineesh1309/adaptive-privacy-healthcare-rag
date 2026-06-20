"""Deterministic direct, quasi and sensitive-attribute detection."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from .models import EntitySpan


@dataclass(frozen=True)
class PatternSpec:
    entity_type: str
    category: str
    pattern: str
    sensitivity: float
    group: int = 0
    flags: int = re.IGNORECASE


PATTERNS = (
    PatternSpec(
        "email",
        "direct",
        r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
        0.98,
    ),
    PatternSpec(
        "phone",
        "direct",
        r"(?<!\d)(?:(?:\+91[-\s]?)?[6-9]\d{9}|(?:\+?1[-\s]?)?\(?\d{3}\)?[-\s.]\d{3}[-\s.]\d{4})(?!\d)",
        0.98,
    ),
    PatternSpec(
        "government_id",
        "direct",
        r"\b(?:Aadhaar|SSN|Social Security(?: Number)?|PAN)\s*(?:number|no\.?|:|#)?\s*([A-Z0-9 -]{8,20})\b",
        1.0,
        group=1,
    ),
    PatternSpec(
        "account_number",
        "direct",
        r"\b(?:account(?: number| no\.?)|a/c)\s*(?:is|:|#|-)?\s*([A-Z0-9-]{6,20})\b",
        1.0,
        group=1,
    ),
    PatternSpec(
        "address",
        "direct",
        r"\b(?:address(?: is)?|lives at|resides at)\s*[:\-]?\s*(\d{1,5}\s+[A-Z0-9][A-Z0-9 .'\-]+?(?:Road|Rd|Street|St|Avenue|Ave|Lane|Ln|Nagar|Layout)\b)",
        0.95,
        group=1,
    ),
    PatternSpec(
        "name",
        "direct",
        r"\b(?:Patient|Customer|Employee|Mr|Mrs|Ms|Dr)\.?\s+(?:(?:is|named)\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\b",
        0.90,
        group=1,
        flags=re.IGNORECASE,
    ),
    PatternSpec(
        "name",
        "direct",
        r"\b([A-Z][a-z]+\s+[A-Z][a-z]+)\s+is\s+(?:an?\s+)?\d{1,3}-year-old\b",
        0.90,
        group=1,
        flags=0,
    ),
    PatternSpec(
        "name",
        "direct",
        r"\b(?:name)\s*(?:is|:)?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\b",
        0.90,
        group=1,
        flags=0,
    ),
    PatternSpec(
        "age",
        "quasi",
        r"\b\d{1,3}(?:-year-old|\s+years?\s+old)\b",
        0.55,
    ),
    PatternSpec(
        "age",
        "quasi",
        r"\bage(?:d)?\s*[: ]\s*(\d{1,3})\b",
        0.55,
        group=1,
    ),
    PatternSpec(
        "date",
        "quasi",
        r"\b(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b",
        0.50,
    ),
    PatternSpec(
        "date",
        "quasi",
        r"\b(?:\d{1,2}(?:st|nd|rd|th)?\s+)?(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)(?:\s+\d{1,2}(?:st|nd|rd|th)?)?(?:,?\s+\d{4})?\b",
        0.50,
    ),
    PatternSpec(
        "location",
        "quasi",
        r"\b(?:from|lives in|based in|located in|city(?: is)?|resides in)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b",
        0.50,
        group=1,
        flags=0,
    ),
    PatternSpec(
        "organization",
        "quasi",
        r"\b(?:[Ww]orks at|[Ee]mployed by|[Cc]ompany|[Oo]rganization)(?:\s+is|\s*:)?\s+([A-Z][A-Za-z&\-]*(?:\s+[A-Z][A-Za-z&\-]*){0,3})\b",
        0.45,
        group=1,
        flags=0,
    ),
    PatternSpec(
        "salary",
        "sensitive",
        r"\b(?:salary|income)\s*(?:is|of|:)?\s*(?:INR|USD|Rs\.?|\$|₹)?\s*[\d,]+(?:\.\d{1,2})?\b",
        0.85,
    ),
    PatternSpec(
        "balance",
        "sensitive",
        r"\b(?:(?:account|available)\s+)?balance\s*(?:was|is|:)?\s*((?:INR|USD|Rs\.?|\$|₹)\s*[\d,]+(?:\.\d{1,2})?)\b",
        0.88,
        group=1,
    ),
    PatternSpec(
        "diagnosis",
        "sensitive",
        r"\b(?:diagnosed with|suffers from|diagnosis(?: is|:)?|treated for)\s+([A-Za-z][A-Za-z \-]{2,40}?)(?=[,.;]|\s+(?:on|and|with|at|in)\b|$)",
        0.82,
        group=1,
    ),
)


class PrivacyDetector:
    """Find privacy spans and remove overlaps deterministically."""

    def __init__(self, patterns: Iterable[PatternSpec] = PATTERNS):
        self.patterns = tuple(patterns)

    def detect(self, text: str) -> list[EntitySpan]:
        candidates: list[EntitySpan] = []
        placeholder_ranges = [
            match.span() for match in re.finditer(r"\[[A-Z][A-Z0-9 -]*\]", text)
        ]
        for spec in self.patterns:
            for match in re.finditer(spec.pattern, text, spec.flags):
                start, end = match.span(spec.group)
                value = text[start:end]
                if not value.strip():
                    continue
                if any(start < right and end > left for left, right in placeholder_ranges):
                    continue
                candidates.append(
                    EntitySpan(
                        entity_type=spec.entity_type,
                        category=spec.category,
                        value=value,
                        start=start,
                        end=end,
                        sensitivity=spec.sensitivity,
                    )
                )

        # Prefer more sensitive and longer spans, then return text order.
        candidates.sort(
            key=lambda item: (-item.sensitivity, -(item.end - item.start), item.start)
        )
        selected: list[EntitySpan] = []
        for candidate in candidates:
            overlaps = any(
                candidate.start < existing.end and candidate.end > existing.start
                for existing in selected
            )
            if not overlaps:
                selected.append(candidate)
        return sorted(selected, key=lambda item: item.start)

    @staticmethod
    def sensitivity_score(entities: Iterable[EntitySpan]) -> float:
        """Aggregate entity risks without allowing a long document to exceed 1."""
        remaining_safety = 1.0
        for entity in entities:
            remaining_safety *= 1.0 - (0.35 * entity.sensitivity)
        return round(min(1.0, 1.0 - remaining_safety), 4)

"""Residual context masking and post-generation leakage checks."""

from __future__ import annotations

from dataclasses import dataclass

from .detector import PrivacyDetector
from .integrity import ContextIntegrityResult, PromptInjectionGuard
from .models import RiskLevel
from .tier_builder import TierBuilder


@dataclass
class GuardResult:
    text: str
    blocked: bool
    detected_count: int


class PrivacyGuard:
    def __init__(self, detector: PrivacyDetector | None = None):
        self.detector = detector or PrivacyDetector()
        self.tier_builder = TierBuilder(self.detector)
        self.injection_guard = PromptInjectionGuard()

    def protect_context(self, text: str, level: RiskLevel | str) -> str:
        # Every tier always receives a fresh direct-identifier scan.
        return self.protect_context_with_trace(text, level).text

    def protect_context_with_trace(
        self, text: str, level: RiskLevel | str
    ) -> ContextIntegrityResult:
        integrity = self.injection_guard.sanitize_context(text)
        return ContextIntegrityResult(
            text=self.tier_builder.sanitize(integrity.text, level),
            injection_detected=integrity.injection_detected,
            categories=integrity.categories,
            removed_segments=integrity.removed_segments,
        )

    def protect_output(self, text: str, level: RiskLevel | str) -> GuardResult:
        selected = RiskLevel.from_value(level)
        entities = self.detector.detect(text)
        direct_count = sum(item.category == "direct" for item in entities)
        if direct_count >= 3:
            return GuardResult(
                text=(
                    "The answer was blocked because it contained multiple residual "
                    "direct identifiers."
                ),
                blocked=True,
                detected_count=len(entities),
            )
        protected = self.tier_builder.sanitize(text, selected, entities)
        return GuardResult(
            text=protected,
            blocked=False,
            detected_count=len(entities),
        )

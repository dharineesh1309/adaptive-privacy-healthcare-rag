"""Residual context masking and post-generation leakage checks."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .detector import PrivacyDetector
from .integrity import ContextIntegrityResult, PromptInjectionGuard
from .models import RiskLevel
from .tier_builder import TierBuilder


# The base paper's Repeat Prompt and ROUGE Prompt thresholds.
COPY_TOKENS = 10
COPY_ROUGE_L = 0.5


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.casefold())


def has_verbatim_run(output: str, sources: list[str], length: int = COPY_TOKENS) -> bool:
    tokens = words(output)
    grams = {tuple(tokens[i : i + length]) for i in range(len(tokens) - length + 1)}
    for source in sources:
        source_tokens = words(source)
        for i in range(len(source_tokens) - length + 1):
            if tuple(source_tokens[i : i + length]) in grams:
                return True
    return False


def rouge_l(candidate: str, reference: str) -> float:
    left, right = words(candidate), words(reference)
    if not left or not right:
        return 0.0
    previous = [0] * (len(right) + 1)
    for token in left:
        current = [0]
        for j, other in enumerate(right, start=1):
            current.append(previous[j - 1] + 1 if token == other else max(previous[j], current[j - 1]))
        previous = current
    common = previous[-1]
    if not common:
        return 0.0
    precision, recall = common / len(left), common / len(right)
    return 2 * precision * recall / (precision + recall)


def copies_context(answer: str, contexts: list[str]) -> bool:
    return has_verbatim_run(answer, contexts) or any(rouge_l(answer, text) > COPY_ROUGE_L for text in contexts)


@dataclass
class GuardResult:
    text: str
    blocked: bool
    detected_count: int


class PrivacyGuard:
    def __init__(self, detector: PrivacyDetector | None = None, block_context_copy: bool = True):
        self.detector = detector or PrivacyDetector()
        self.tier_builder = TierBuilder(self.detector)
        self.injection_guard = PromptInjectionGuard()
        self.block_context_copy = block_context_copy

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

    def protect_output(
        self, text: str, level: RiskLevel | str, contexts: list[str] | None = None
    ) -> GuardResult:
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
        # Reacts to what the answer contains, however the request was worded.
        if self.block_context_copy and contexts and copies_context(text, contexts):
            return GuardResult(
                text="The answer was blocked because it copied text from the retrieved records.",
                blocked=True,
                detected_count=len(entities),
            )
        protected = self.tier_builder.sanitize(text, selected, entities)
        return GuardResult(
            text=protected,
            blocked=False,
            detected_count=len(entities),
        )

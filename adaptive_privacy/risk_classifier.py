"""Explainable, rule-based query-risk classification."""

from __future__ import annotations

import re

from .integrity import PromptInjectionGuard
from .models import QueryRiskDecision, RiskLevel


class QueryRiskClassifier:
    def __init__(self, config: dict):
        self.config = config
        self.risk_config = config["risk"]
        self.role_scores = {
            key.lower(): float(value)
            for key, value in config.get("roles", {}).items()
        }
        self.injection_guard = PromptInjectionGuard()

    def classify(self, query: str, role: str = "unknown") -> QueryRiskDecision:
        normalized = " ".join(query.lower().split())
        score = float(self.risk_config.get("base_score", 0.1))
        reasons: list[str] = []

        forced_matches = self._matches(
            normalized, self.risk_config.get("forced_high_patterns", [])
        )
        injection = self.injection_guard.assess(normalized)
        forced_high = bool(forced_matches) or injection.detected
        if forced_matches:
            reasons.append(f"Extraction pattern: {', '.join(forced_matches)}")
        if injection.detected:
            reasons.append(
                "Prompt-injection signal: " + ", ".join(injection.categories)
            )

        for group in self.risk_config.get("pattern_groups", []):
            matches = self._matches(normalized, group.get("terms", []))
            if matches:
                score += float(group.get("weight", 0.0))
                reasons.append(f"{group['label']}: {', '.join(matches)}")

        score = round(max(0.0, min(1.0, score)), 4)
        role_name = role.lower().strip() or "unknown"
        role_score = self.role_scores.get(
            role_name, self.role_scores.get("unknown", 1.0)
        )

        if forced_high:
            level = RiskLevel.HIGH
            score = 1.0
        else:
            level = self.level_for_score(score)

        if not reasons:
            reasons.append("No sensitive query indicators detected")
        if role_name not in self.role_scores:
            reasons.append("Unknown role treated as public")

        return QueryRiskDecision(
            query_score=score,
            level=level,
            role=role_name,
            role_score=role_score,
            reasons=reasons,
            forced_high=forced_high,
        )

    def level_for_score(self, score: float) -> RiskLevel:
        thresholds = self.risk_config["thresholds"]
        if score < float(thresholds["low_max"]):
            return RiskLevel.LOW
        if score < float(thresholds["high_min"]):
            return RiskLevel.MEDIUM
        return RiskLevel.HIGH

    def retrieval_query(self, query: str) -> str:
        """Remove attack/control phrases that would distort document retrieval."""
        cleaned = query
        phrases = list(self.risk_config.get("forced_high_patterns", []))
        phrases.extend(self.injection_guard.assess(query).matches)
        for group in self.risk_config.get("pattern_groups", []):
            if float(group.get("weight", 0.0)) >= 0.5:
                phrases.extend(group.get("terms", []))
        for phrase in sorted(phrases, key=len, reverse=True):
            cleaned = re.sub(re.escape(phrase), " ", cleaned, flags=re.IGNORECASE)
        cleaned = " ".join(cleaned.split())
        return cleaned or query

    @staticmethod
    def _matches(text: str, terms: list[str]) -> list[str]:
        return [term for term in terms if term.lower() in text]

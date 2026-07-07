"""Fuse query, document and role risk into a privacy-tier decision."""

from __future__ import annotations

from .models import PolicyDecision, QueryRiskDecision, RiskLevel


class PolicyEngine:
    def __init__(self, config: dict):
        self.risk_config = config["risk"]

    def decide(
        self,
        document_id: str,
        document_score: float,
        query_decision: QueryRiskDecision,
        forced_level: RiskLevel | str | None = None,
        minimum_level: RiskLevel | str | None = None,
    ) -> PolicyDecision:
        weights = self.risk_config["weights"]
        final_score = (
            float(weights["query"]) * query_decision.query_score
            + float(weights["document"]) * document_score
            + float(weights["role"]) * query_decision.role_score
        )
        final_score = round(max(0.0, min(1.0, final_score)), 4)
        reasons = [
            f"query={query_decision.query_score:.2f}",
            f"document={document_score:.2f}",
            f"role={query_decision.role_score:.2f}",
        ]

        if forced_level is not None:
            selected = RiskLevel.from_value(forced_level)
            reasons.append(f"Evaluation mode forced {selected.value} tier")
        elif query_decision.forced_high:
            selected = RiskLevel.HIGH
            reasons.append("Attack-pattern override forced high tier")
        else:
            selected = self._level_for_score(final_score)

        if minimum_level is not None:
            minimum = RiskLevel.from_value(minimum_level)
            order = {RiskLevel.LOW: 0, RiskLevel.MEDIUM: 1, RiskLevel.HIGH: 2}
            if order[selected] < order[minimum]:
                selected = minimum
                reasons.append(f"Session policy raised minimum tier to {minimum.value}")

        return PolicyDecision(
            document_id=str(document_id),
            final_score=final_score,
            selected_tier=selected,
            query_score=query_decision.query_score,
            document_score=document_score,
            role_score=query_decision.role_score,
            reasons=reasons,
        )

    def _level_for_score(self, score: float) -> RiskLevel:
        thresholds = self.risk_config["thresholds"]
        if score < float(thresholds["low_max"]):
            return RiskLevel.LOW
        if score < float(thresholds["high_min"]):
            return RiskLevel.MEDIUM
        return RiskLevel.HIGH

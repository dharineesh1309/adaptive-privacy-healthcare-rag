"""Purpose-bound contextual-integrity policy for disclosure decisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .models import RiskLevel


@dataclass(frozen=True)
class PurposeDecision:
    purpose: str
    role: str
    allowed: bool
    requested_types: list[str]
    disallowed_types: list[str]
    reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "purpose": self.purpose,
            "role": self.role,
            "allowed": self.allowed,
            "requested_types": self.requested_types,
            "disallowed_types": self.disallowed_types,
            "reasons": self.reasons,
        }


class PurposePolicy:
    def __init__(self, config: dict[str, Any]):
        self.config = config.get("purpose", {})
        self.rules = self.config.get("rules", {})
        self.default = str(self.config.get("default", "general_information"))

    @property
    def allowed_purposes(self) -> set[str]:
        return set(self.rules)

    def evaluate(
        self,
        purpose: str,
        role: str,
        requested_types: set[str],
        query_level: RiskLevel,
    ) -> PurposeDecision:
        selected = purpose.strip().lower() or self.default
        if selected not in self.rules:
            raise ValueError(f"Unsupported purpose: {selected}")
        rule = self.rules[selected]
        roles = {str(item).lower() for item in rule.get("roles", [])}
        allowed_types = {str(item) for item in rule.get("allowed_types", [])}
        reasons: list[str] = []
        role_allowed = role in roles
        if role_allowed:
            reasons.append(f"Role {role} is permitted for {selected}")
        else:
            reasons.append(f"Role {role} is not permitted for {selected}")

        enforce_scope = bool(rule.get("enforce_attribute_scope", True))
        if selected == "general_information" and query_level is RiskLevel.LOW:
            enforce_scope = False
            reasons.append("Low-risk educational intent does not claim a private record")
        disallowed = sorted(requested_types - allowed_types) if enforce_scope else []
        if disallowed:
            reasons.append(
                "Purpose does not permit requested attributes: "
                + ", ".join(disallowed)
            )
        allowed = role_allowed and not disallowed
        if allowed:
            reasons.append("Role-purpose-attribute flow is permitted")
        else:
            reasons.append("Contextual-integrity policy denied the information flow")
        return PurposeDecision(
            purpose=selected,
            role=role,
            allowed=allowed,
            requested_types=sorted(requested_types),
            disallowed_types=disallowed,
            reasons=reasons,
        )

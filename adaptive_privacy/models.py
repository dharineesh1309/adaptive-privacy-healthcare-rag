"""Data models shared by the adaptive privacy pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

    @classmethod
    def from_value(cls, value: str | "RiskLevel") -> "RiskLevel":
        if isinstance(value, cls):
            return value
        return cls(str(value).lower())


@dataclass(frozen=True)
class EntitySpan:
    entity_type: str
    category: str
    value: str
    start: int
    end: int
    sensitivity: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DocumentVariants:
    document_id: str
    sensitivity_score: float
    low_text: str
    medium_text: str
    high_text: str
    detected_entities: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def select(self, level: RiskLevel | str) -> str:
        selected = RiskLevel.from_value(level)
        return {
            RiskLevel.LOW: self.low_text,
            RiskLevel.MEDIUM: self.medium_text,
            RiskLevel.HIGH: self.high_text,
        }[selected]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "DocumentVariants":
        return cls(
            document_id=str(value["document_id"]),
            sensitivity_score=float(value["sensitivity_score"]),
            low_text=str(value["low_text"]),
            medium_text=str(value["medium_text"]),
            high_text=str(value["high_text"]),
            detected_entities=list(value.get("detected_entities", [])),
            metadata=dict(value.get("metadata", {})),
        )


@dataclass
class QueryRiskDecision:
    query_score: float
    level: RiskLevel
    role: str
    role_score: float
    reasons: list[str] = field(default_factory=list)
    forced_high: bool = False

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["level"] = self.level.value
        return value


@dataclass
class PolicyDecision:
    document_id: str
    final_score: float
    selected_tier: RiskLevel
    query_score: float
    document_score: float
    role_score: float
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["selected_tier"] = self.selected_tier.value
        return value


@dataclass
class RetrievalResult:
    document_id: str
    score: float
    document: DocumentVariants


@dataclass
class PipelineResult:
    query: str
    role: str
    mode: str
    query_decision: QueryRiskDecision
    policy_decisions: list[PolicyDecision]
    protected_contexts: list[str]
    answer: str
    blocked: bool
    latency_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "role": self.role,
            "mode": self.mode,
            "query_decision": self.query_decision.to_dict(),
            "policy_decisions": [item.to_dict() for item in self.policy_decisions],
            "protected_contexts": self.protected_contexts,
            "answer": self.answer,
            "blocked": self.blocked,
            "latency_ms": round(self.latency_ms, 3),
        }

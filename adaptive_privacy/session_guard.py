"""Session-aware cumulative privacy risk and multi-turn policy escalation."""

from __future__ import annotations

import re
import threading
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from .models import DocumentVariants, QueryRiskDecision, RiskLevel


# Request-shaped phrases: patients write "my name is", "loss of balance" or "pay for
# tests" without asking for anyone's identifier, so bare words are not used for these.
REQUEST_PATTERNS = {
    "phone": ("phone number", "mobile number", "contact number", "phone no"),
    "email": ("email address", "email id", "e-mail address"),
    "address": ("home address", "address of", "their address", "his address", "her address",
                "where does", "where do they live"),
    "account_number": ("account number", "a/c"),
    "name": ("name of the patient", "name of this patient", "patient's name", "patient name",
             "full name", "their name", "his name", "her name", "who is the patient"),
    "age": ("age", "how old", "age group"),
    "location": ("city", "location", "where from", "region"),
    "date": ("date", "when was", "what year"),
    "organization": ("which company", "what company", "organization", "employer", "works at"),
    "diagnosis": ("diagnosis", "condition", "disease"),
    "salary": ("salary", "income", "paycheck"),
    "balance": ("account balance", "bank balance", "balance of", "balance in"),
    "treatment": ("treatment", "medication", "prescribed"),
}

DIRECT_TYPES = {"phone", "email", "address", "account_number", "name"}
QUASI_TYPES = {"age", "location", "date", "organization"}
SENSITIVE_TYPES = {"diagnosis", "salary", "balance", "treatment"}
LEVEL_ORDER = {RiskLevel.LOW: 0, RiskLevel.MEDIUM: 1, RiskLevel.HIGH: 2}


def infer_requested_types(query: str) -> set[str]:
    normalized_query = " ".join(query.lower().split())
    # Whole words only (optional plural): "age" must not match "average"/"manage".
    # A first-person "my ..." is self-disclosure, not a request for someone's value.
    return {
        entity_type
        for entity_type, patterns in REQUEST_PATTERNS.items()
        if any(
            re.search(rf"(?<!\bmy )\b{re.escape(pattern)}s?\b", normalized_query)
            for pattern in patterns
        )
    }


@dataclass
class SessionTurn:
    turn: int
    action: str
    cumulative_score: float
    query_level: str
    requested_types: list[str]
    document_ids: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn": self.turn,
            "action": self.action,
            "cumulative_score": self.cumulative_score,
            "query_level": self.query_level,
            "requested_types": self.requested_types,
            "document_ids": self.document_ids,
        }


@dataclass
class SessionState:
    session_id: str
    role: str | None = None
    turn_count: int = 0
    cumulative_score: float = 0.0
    blocked_attempts: int = 0
    document_access: Counter[str] = field(default_factory=Counter)
    requested_types: set[str] = field(default_factory=set)
    exposed_types: set[str] = field(default_factory=set)
    history: list[SessionTurn] = field(default_factory=list)
    updated_at: float = field(default_factory=time.time)


@dataclass
class SessionRiskDecision:
    session_id: str
    turn: int
    action: str
    minimum_tier: RiskLevel
    cumulative_score: float
    reasons: list[str]
    requested_types: list[str]
    repeated_documents: list[str]
    role_changed: bool
    unverified_role_claim: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "turn": self.turn,
            "action": self.action,
            "minimum_tier": self.minimum_tier.value,
            "cumulative_score": self.cumulative_score,
            "reasons": self.reasons,
            "requested_types": self.requested_types,
            "repeated_documents": self.repeated_documents,
            "role_changed": self.role_changed,
            "unverified_role_claim": self.unverified_role_claim,
        }


class SessionRiskManager:
    def __init__(self, config: dict):
        self.config = config.get("session", {})
        self.states: dict[str, SessionState] = {}
        self.lock = threading.RLock()

    def create(self) -> dict[str, Any]:
        session_id = str(uuid.uuid4())
        with self.lock:
            self.states[session_id] = SessionState(session_id=session_id)
        return self.snapshot(session_id)

    def reset(self, session_id: str) -> dict[str, Any]:
        with self.lock:
            if session_id not in self.states:
                raise KeyError("Unknown session.")
            self.states[session_id] = SessionState(session_id=session_id)
        return self.snapshot(session_id)

    def evaluate(
        self,
        session_id: str,
        query: str,
        role: str,
        query_decision: QueryRiskDecision,
        document_ids: list[str],
        proposed_tiers: list[RiskLevel],
        unverified_role_claim: bool = False,
    ) -> SessionRiskDecision:
        with self.lock:
            state = self._state(session_id)
            session_config = self.config
            signals = session_config.get("signals", {})
            thresholds = session_config.get("thresholds", {})
            decay = float(session_config.get("decay", 0.72))

            requested = infer_requested_types(query)
            new_types = requested - state.requested_types
            all_requested = state.requested_types | requested
            repeated = sorted(
                document_id
                for document_id in set(document_ids)
                if state.document_access[document_id] > 0
            )
            role_changed = bool(
                state.role is not None and state.role != role and state.turn_count > 0
            )

            increment = float(signals.get("query_risk", 0.22)) * query_decision.query_score
            reasons = [f"Current query risk contributes {increment:.2f}"]
            if new_types:
                addition = float(signals.get("new_attribute", 0.08)) * len(new_types)
                increment += addition
                reasons.append(
                    f"New requested attributes: {', '.join(sorted(new_types))}"
                )
            if repeated:
                addition = float(signals.get("repeated_document", 0.12)) * min(
                    len(repeated), 2
                )
                increment += addition
                reasons.append(f"Repeated access to {len(repeated)} document(s)")
            diversity_count = max(0, len(all_requested) - 2)
            if diversity_count:
                addition = float(signals.get("attribute_diversity", 0.06)) * diversity_count
                increment += addition
                reasons.append(
                    f"Cumulative attribute diversity reached {len(all_requested)} types"
                )
            if query_decision.forced_high:
                increment += float(signals.get("attack_pattern", 0.35))
                reasons.append("Extraction-attack pattern detected")
            if role_changed:
                increment += float(signals.get("role_change", 0.30))
                reasons.append(f"Role changed from {state.role} to {role}")
            if unverified_role_claim:
                increment += float(signals.get("unverified_role_claim", 0.25))
                reasons.append("Unverified privileged-role claim")
            if state.blocked_attempts:
                increment += float(signals.get("prior_block", 0.20))
                reasons.append("Session contains a prior blocked attempt")

            cumulative = min(1.0, state.cumulative_score * decay + increment)
            block_threshold = float(thresholds.get("block", 0.85))
            high_threshold = float(thresholds.get("high", 0.55))
            medium_threshold = float(thresholds.get("medium", 0.30))
            if cumulative >= block_threshold:
                minimum = RiskLevel.HIGH
                action = "block"
                reasons.append("Cumulative privacy budget exhausted")
            elif cumulative >= high_threshold or query_decision.forced_high or role_changed:
                minimum = RiskLevel.HIGH
                action = "escalate"
            elif cumulative >= medium_threshold:
                minimum = RiskLevel.MEDIUM
                action = "escalate"
            else:
                minimum = RiskLevel.LOW
                action = "allow"

            if action == "escalate" and proposed_tiers:
                if all(
                    LEVEL_ORDER[tier] >= LEVEL_ORDER[minimum]
                    for tier in proposed_tiers
                ):
                    action = "allow"
                    reasons.append("Base policy already satisfies session minimum")

            state.turn_count += 1
            state.cumulative_score = round(cumulative, 4)
            state.role = role
            state.updated_at = time.time()
            state.requested_types.update(requested)
            state.document_access.update(document_ids)
            if action == "block":
                state.blocked_attempts += 1
            turn = SessionTurn(
                turn=state.turn_count,
                action=action,
                cumulative_score=state.cumulative_score,
                query_level=query_decision.level.value,
                requested_types=sorted(requested),
                document_ids=list(document_ids),
            )
            state.history.append(turn)
            max_history = int(session_config.get("max_history", 20))
            state.history = state.history[-max_history:]

            return SessionRiskDecision(
                session_id=session_id,
                turn=state.turn_count,
                action=action,
                minimum_tier=minimum,
                cumulative_score=state.cumulative_score,
                reasons=reasons,
                requested_types=sorted(requested),
                repeated_documents=repeated,
                role_changed=role_changed,
                unverified_role_claim=unverified_role_claim,
            )

    def record_exposure(
        self,
        session_id: str,
        documents: dict[str, DocumentVariants],
        tiers: dict[str, RiskLevel],
    ) -> None:
        with self.lock:
            state = self._state(session_id)
            for document_id, tier in tiers.items():
                document = documents.get(document_id)
                if document is None:
                    continue
                for entity in document.detected_entities:
                    category = entity.get("category")
                    entity_type = str(entity.get("entity_type", "unknown"))
                    if tier is RiskLevel.LOW and category in {"quasi", "sensitive"}:
                        state.exposed_types.add(entity_type)
                    elif tier is RiskLevel.MEDIUM and category == "sensitive":
                        state.exposed_types.add(entity_type)
            state.updated_at = time.time()

    def snapshot(self, session_id: str) -> dict[str, Any]:
        with self.lock:
            state = self._state(session_id)
            quasi_used = len(state.exposed_types & QUASI_TYPES)
            sensitive_used = len(state.exposed_types & SENSITIVE_TYPES)
            budgets = self.config.get("budgets", {})
            quasi_limit = int(budgets.get("quasi_types", 3))
            sensitive_limit = int(budgets.get("sensitive_types", 2))
            return {
                "session_id": state.session_id,
                "turn_count": state.turn_count,
                "cumulative_score": state.cumulative_score,
                "blocked_attempts": state.blocked_attempts,
                "role": state.role,
                "requested_types": sorted(state.requested_types),
                "exposed_types": sorted(state.exposed_types),
                "remaining_budget": {
                    "quasi_types": max(0, quasi_limit - quasi_used),
                    "sensitive_types": max(0, sensitive_limit - sensitive_used),
                },
                "history": [item.to_dict() for item in state.history],
            }

    def _state(self, session_id: str) -> SessionState:
        try:
            return self.states[session_id]
        except KeyError as error:
            raise KeyError("Unknown session.") from error

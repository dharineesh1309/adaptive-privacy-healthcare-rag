"""Wrapper that applies cumulative session policy before serving contexts."""

from __future__ import annotations

import time
from typing import Any

from .models import RiskLevel
from .pipeline import AdaptivePrivacyPipeline
from .privacy_ledger import PersistentPrivacyLedger
from .purpose_policy import PurposePolicy
from .session_guard import SessionRiskManager, infer_requested_types


LEVEL_ORDER = {RiskLevel.LOW: 0, RiskLevel.MEDIUM: 1, RiskLevel.HIGH: 2}


class SessionAwarePrivacyPipeline:
    def __init__(
        self,
        pipeline: AdaptivePrivacyPipeline,
        manager: SessionRiskManager | None = None,
        ledger: PersistentPrivacyLedger | None = None,
    ):
        self.pipeline = pipeline
        self.manager = manager or SessionRiskManager(pipeline.config)
        self.ledger = ledger or PersistentPrivacyLedger(pipeline.config)
        self.purpose_policy = PurposePolicy(pipeline.config)
        self.documents = {item.document_id: item for item in pipeline.documents}

    def create_session(self, principal_id: str = "lab-user-01") -> dict[str, Any]:
        session = self.manager.create()
        session["ledger"] = self.ledger.snapshot(principal_id)
        return session

    def reset_session(
        self, session_id: str, principal_id: str = "lab-user-01"
    ) -> dict[str, Any]:
        session = self.manager.reset(session_id)
        session["ledger"] = self.ledger.snapshot(principal_id)
        return session

    def reset_ledger(self, principal_id: str) -> dict[str, Any]:
        return self.ledger.reset(principal_id)

    def process(
        self,
        session_id: str,
        query: str,
        role: str,
        mode: str = "adaptive",
        role_verified: bool = False,
        purpose: str = "general_information",
        principal_id: str = "lab-user-01",
        top_k: int | None = None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        role_decision = self.pipeline.role_assurance.resolve(role, role_verified)
        query_decision = self.pipeline.classifier.classify(
            query, role_decision.effective_role
        )
        if role_decision.downgraded:
            query_decision.reasons.append(role_decision.reason)
        requested_types = infer_requested_types(query)
        purpose_decision = self.purpose_policy.evaluate(
            purpose=purpose,
            role=role_decision.effective_role,
            requested_types=requested_types,
            query_level=query_decision.level,
        )
        retrieval_query = self.pipeline.classifier.retrieval_query(query)
        retrieval_top_k = top_k or int(
            self.pipeline.config.get("retrieval", {}).get("top_k", 3)
        )
        retrieved = self.pipeline.retriever.retrieve(retrieval_query, retrieval_top_k)
        forced_level = {
            "fixed_low": RiskLevel.LOW,
            "fixed_high": RiskLevel.HIGH,
        }.get(mode)
        proposed = [
            self.pipeline.policy_engine.decide(
                item.document_id,
                item.document.sensitivity_score,
                query_decision,
                forced_level=forced_level,
            )
            for item in retrieved
        ]
        session_decision = self.manager.evaluate(
            session_id=session_id,
            query=query,
            role=role_decision.effective_role,
            query_decision=query_decision,
            document_ids=[item.document_id for item in retrieved],
            proposed_tiers=[item.selected_tier for item in proposed],
            unverified_role_claim=role_decision.downgraded,
        )

        session_adjusted = [
            self.pipeline.policy_engine.decide(
                item.document_id,
                item.document.sensitivity_score,
                query_decision,
                forced_level=forced_level,
                minimum_level=session_decision.minimum_tier,
            )
            for item in retrieved
        ]
        ledger_decision = self.ledger.preview(
            principal_id=principal_id,
            documents=self.documents,
            tiers={item.document_id: item.selected_tier for item in session_adjusted},
            requested_types=requested_types,
        )
        minimum_tier = max(
            (session_decision.minimum_tier, ledger_decision.minimum_tier),
            key=lambda level: LEVEL_ORDER[level],
        )
        blocked_by = []
        # A detected "repeat/print the context" request is refused, not served masked:
        # even the high tier keeps the record's narrative verbatim.
        if self.pipeline.config.get("integrity", {}).get(
            "block_context_exfiltration", True
        ) and "context_exfiltration" in (
            self.pipeline.classifier.injection_guard.assess(query).categories
        ):
            blocked_by.append("context exfiltration guard")
        if not purpose_decision.allowed:
            blocked_by.append("purpose policy")
        if session_decision.action == "block":
            blocked_by.append("session budget")
        if ledger_decision.action == "block":
            blocked_by.append("persistent principal budget")

        if blocked_by:
            blocked_decisions = [
                self.pipeline.policy_engine.decide(
                    item.document_id,
                    item.document.sensitivity_score,
                    query_decision,
                    forced_level=forced_level,
                    minimum_level=RiskLevel.HIGH,
                )
                for item in retrieved
            ]
            response = {
                "query": query,
                "role": role,
                "mode": mode,
                "query_decision": query_decision.to_dict(),
                "policy_decisions": [item.to_dict() for item in blocked_decisions],
                "protected_contexts": [],
                "answer": (
                    "This request was blocked by the "
                    + ", ".join(blocked_by)
                    + ". Request information consistent with the verified role, "
                    "declared purpose, and remaining disclosure budget."
                ),
                "blocked": True,
                "latency_ms": round((time.perf_counter() - started) * 1000.0, 3),
                "role_assurance": role_decision.to_dict(),
                "security_events": [],
                "generation": {
                    **self.pipeline.generation_status(),
                    "mode": "blocked_before_generation",
                    "called": False,
                    "cached": False,
                    "reason": "request blocked by deterministic privacy policy",
                },
            }
            audit_event = dict(response)
            audit_event["session"] = session_decision.to_dict()
            audit_event["purpose_decision"] = purpose_decision.to_dict()
            audit_event["ledger_decision"] = ledger_decision.to_dict()
            self.pipeline.audit.write(audit_event)
        else:
            result = self.pipeline.process(
                query=query,
                role=role,
                mode=mode,
                minimum_tier=minimum_tier,
                role_verified=role_verified,
                top_k=retrieval_top_k,
            )
            response = result.to_dict()
            tiers = {
                item.document_id: item.selected_tier
                for item in result.policy_decisions
            }
            self.manager.record_exposure(session_id, self.documents, tiers)
            self.ledger.record_exposure(
                principal_id=principal_id,
                purpose=purpose_decision.purpose,
                documents=self.documents,
                tiers=tiers,
            )

        response["session_decision"] = session_decision.to_dict()
        response["session"] = self.manager.snapshot(session_id)
        response["purpose_decision"] = purpose_decision.to_dict()
        response["ledger_decision"] = ledger_decision.to_dict()
        response["ledger"] = self.ledger.snapshot(principal_id)
        return response

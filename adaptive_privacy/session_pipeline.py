"""Wrapper that applies cumulative session policy before serving contexts."""

from __future__ import annotations

import time
from typing import Any

from .models import RiskLevel
from .pipeline import AdaptivePrivacyPipeline
from .session_guard import SessionRiskManager


class SessionAwarePrivacyPipeline:
    def __init__(
        self,
        pipeline: AdaptivePrivacyPipeline,
        manager: SessionRiskManager | None = None,
    ):
        self.pipeline = pipeline
        self.manager = manager or SessionRiskManager(pipeline.config)
        self.documents = {item.document_id: item for item in pipeline.documents}

    def create_session(self) -> dict[str, Any]:
        return self.manager.create()

    def reset_session(self, session_id: str) -> dict[str, Any]:
        return self.manager.reset(session_id)

    def process(
        self,
        session_id: str,
        query: str,
        role: str,
        mode: str = "adaptive",
        role_verified: bool = False,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        role_decision = self.pipeline.role_assurance.resolve(role, role_verified)
        query_decision = self.pipeline.classifier.classify(
            query, role_decision.effective_role
        )
        if role_decision.downgraded:
            query_decision.reasons.append(role_decision.reason)
        retrieval_query = self.pipeline.classifier.retrieval_query(query)
        top_k = int(self.pipeline.config.get("retrieval", {}).get("top_k", 3))
        retrieved = self.pipeline.retriever.retrieve(retrieval_query, top_k)
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

        if session_decision.action == "block":
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
                    "This request was blocked because the session's cumulative "
                    "privacy budget was exhausted. Start a new authorized session "
                    "or request non-identifying aggregate information."
                ),
                "blocked": True,
                "latency_ms": round((time.perf_counter() - started) * 1000.0, 3),
                "role_assurance": role_decision.to_dict(),
                "security_events": [],
            }
            audit_event = dict(response)
            audit_event["session"] = session_decision.to_dict()
            self.pipeline.audit.write(audit_event)
        else:
            result = self.pipeline.process(
                query=query,
                role=role,
                mode=mode,
                minimum_tier=session_decision.minimum_tier,
                role_verified=role_verified,
            )
            response = result.to_dict()
            tiers = {
                item.document_id: item.selected_tier
                for item in result.policy_decisions
            }
            self.manager.record_exposure(session_id, self.documents, tiers)

        response["session_decision"] = session_decision.to_dict()
        response["session"] = self.manager.snapshot(session_id)
        return response

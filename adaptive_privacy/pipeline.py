"""End-to-end adaptive privacy pipeline used by the demo and evaluation."""

from __future__ import annotations

import re
import time
from collections.abc import Callable

from .audit import AuditLogger
from .config import load_config
from .guards import PrivacyGuard
from .models import DocumentVariants, PipelineResult, RiskLevel
from .policy_engine import PolicyEngine
from .retriever import BM25Retriever, tokenize
from .risk_classifier import QueryRiskClassifier
from .role_assurance import RoleAssurancePolicy


Generator = Callable[[str, list[str]], str]


class AdaptivePrivacyPipeline:
    def __init__(
        self,
        documents: list[DocumentVariants],
        config: dict | None = None,
        generator: Generator | None = None,
        audit_path: str | None = None,
    ):
        self.config = config or load_config()
        self.documents = documents
        self.classifier = QueryRiskClassifier(self.config)
        self.role_assurance = RoleAssurancePolicy()
        self.policy_engine = PolicyEngine(self.config, documents)
        self.retriever = BM25Retriever(documents)
        self.guard = PrivacyGuard(
            block_context_copy=self.config.get("integrity", {}).get("block_context_copy", True)
        )
        self.generator = generator or extractive_generator
        self.audit = AuditLogger(audit_path)

    def process(
        self,
        query: str,
        role: str = "unknown",
        top_k: int | None = None,
        mode: str = "adaptive",
        minimum_tier: RiskLevel | str | None = None,
        role_verified: bool = False,
        use_generator: bool = True,
    ) -> PipelineResult:
        if mode not in {"adaptive", "fixed_low", "fixed_high"}:
            raise ValueError("mode must be adaptive, fixed_low or fixed_high")
        started = time.perf_counter()
        role_decision = self.role_assurance.resolve(role, role_verified)
        query_decision = self.classifier.classify(query, role_decision.effective_role)
        if role_decision.downgraded:
            query_decision.reasons.append(role_decision.reason)
        retrieval_limit = top_k or int(self.config.get("retrieval", {}).get("top_k", 3))
        retrieval_query = self.classifier.retrieval_query(query)
        retrieved = self.retriever.retrieve(retrieval_query, retrieval_limit)

        forced_level = {
            "fixed_low": RiskLevel.LOW,
            "fixed_high": RiskLevel.HIGH,
        }.get(mode)
        decisions = []
        contexts = []
        security_events = []
        for item in retrieved:
            decision = self.policy_engine.decide(
                document_id=item.document_id,
                document_score=item.document.sensitivity_score,
                query_decision=query_decision,
                forced_level=forced_level,
                minimum_level=minimum_tier,
            )
            selected_text = item.document.select(decision.selected_tier)
            context_result = self.guard.protect_context_with_trace(
                selected_text, decision.selected_tier
            )
            contexts.append(context_result.text)
            if context_result.injection_detected:
                security_events.append(
                    {
                        "type": "retrieved_prompt_injection",
                        "document_id": item.document_id,
                        "categories": context_result.categories,
                        "removed_segments": context_result.removed_segments,
                    }
                )
            decisions.append(decision)

        if contexts and use_generator:
            draft_answer = self.generator(query, contexts)
            generation = self._last_generation_metadata()
        elif contexts:
            draft_answer = extractive_generator(query, contexts)
            generation = self.generation_status()
            generation.update(
                {
                    "mode": "extractive_evaluation",
                    "called": False,
                    "cached": False,
                    "reason": "external generation disabled for policy comparison",
                }
            )
        else:
            draft_answer = "No relevant protected context was found."
            generation = self.generation_status()
            generation.update(
                {
                    "mode": "no_context",
                    "called": False,
                    "cached": False,
                    "reason": "no protected context reached generation",
                }
            )

        strictest_level = max(
            (decision.selected_tier for decision in decisions),
            default=query_decision.level,
            key=lambda level: list(RiskLevel).index(level),
        )
        # Extractive answers copy context by design, and copying a public record
        # (no detected private values) is harmless; only those cases go unchecked.
        model_written = not str(generation.get("mode", "")).startswith(("extractive", "no_context"))
        private_contexts = [
            text for text, item in zip(contexts, retrieved) if item.document.detected_entities
        ]
        guarded = self.guard.protect_output(
            draft_answer, strictest_level, private_contexts if model_written else None
        )
        latency_ms = (time.perf_counter() - started) * 1000.0
        result = PipelineResult(
            query=query,
            role=role,
            mode=mode,
            query_decision=query_decision,
            policy_decisions=decisions,
            protected_contexts=contexts,
            answer=guarded.text,
            blocked=guarded.blocked,
            latency_ms=latency_ms,
            role_assurance=role_decision.to_dict(),
            security_events=security_events,
            generation=generation,
        )
        audit_event = result.to_dict()
        self.audit.write(audit_event)
        return result

    def generation_status(self) -> dict:
        status = getattr(self.generator, "status", None)
        if callable(status):
            return dict(status())
        if self.generator is extractive_generator:
            return {
                "provider": "local",
                "model": "extractive",
                "configured": True,
                "mode": "extractive",
                "called": False,
                "cached": False,
                "reason": "",
            }
        return {
            "provider": "custom",
            "model": getattr(self.generator, "__name__", "custom"),
            "configured": True,
            "mode": "custom",
            "called": True,
            "cached": False,
            "reason": "",
        }

    def _last_generation_metadata(self) -> dict:
        metadata = getattr(self.generator, "last_metadata", None)
        return dict(metadata()) if callable(metadata) else self.generation_status()


def extractive_generator(query: str, contexts: list[str]) -> str:
    """Dependency-free demo generator; replace with an LLM adapter later.

    The demo intentionally answers from only the highest-ranked protected record.
    This avoids combining attributes belonging to different people.
    """
    broad_subject_terms = {"patient", "customer", "employee", "person"}
    query_terms = set(tokenize(query)) - broad_subject_terms
    candidates: list[tuple[int, str]] = []
    for sentence in re.split(r"(?<=[.!?])\s+", contexts[0]):
        cleaned = sentence.strip()
        if not cleaned:
            continue
        overlap = len(query_terms.intersection(tokenize(cleaned)))
        candidates.append((overlap, cleaned))
    candidates.sort(key=lambda item: (-item[0], -len(item[1])))
    selected = [sentence for _, sentence in candidates[:2]]
    return " ".join(selected) if selected else "No supported answer was found."

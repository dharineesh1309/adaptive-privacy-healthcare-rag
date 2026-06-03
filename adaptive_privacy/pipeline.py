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
        self.policy_engine = PolicyEngine(self.config)
        self.retriever = BM25Retriever(documents)
        self.guard = PrivacyGuard()
        self.generator = generator or extractive_generator
        self.audit = AuditLogger(audit_path)

    def process(
        self,
        query: str,
        role: str = "unknown",
        top_k: int | None = None,
        mode: str = "adaptive",
    ) -> PipelineResult:
        if mode not in {"adaptive", "fixed_low", "fixed_high"}:
            raise ValueError("mode must be adaptive, fixed_low or fixed_high")
        started = time.perf_counter()
        query_decision = self.classifier.classify(query, role)
        retrieval_limit = top_k or int(self.config.get("retrieval", {}).get("top_k", 3))
        retrieved = self.retriever.retrieve(query, retrieval_limit)

        forced_level = {
            "fixed_low": RiskLevel.LOW,
            "fixed_high": RiskLevel.HIGH,
        }.get(mode)
        decisions = []
        contexts = []
        for item in retrieved:
            decision = self.policy_engine.decide(
                document_id=item.document_id,
                document_score=item.document.sensitivity_score,
                query_decision=query_decision,
                forced_level=forced_level,
            )
            selected_text = item.document.select(decision.selected_tier)
            contexts.append(
                self.guard.protect_context(selected_text, decision.selected_tier)
            )
            decisions.append(decision)

        if contexts:
            draft_answer = self.generator(query, contexts)
        else:
            draft_answer = "No relevant protected context was found."

        strictest_level = max(
            (decision.selected_tier for decision in decisions),
            default=query_decision.level,
            key=lambda level: list(RiskLevel).index(level),
        )
        guarded = self.guard.protect_output(draft_answer, strictest_level)
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
        )
        self.audit.write(result.to_dict())
        return result


def extractive_generator(query: str, contexts: list[str]) -> str:
    """Dependency-free demo generator; replace with an LLM adapter later."""
    query_terms = set(tokenize(query))
    candidates: list[tuple[int, str]] = []
    for context in contexts:
        for sentence in re.split(r"(?<=[.!?])\s+", context):
            cleaned = sentence.strip()
            if not cleaned:
                continue
            overlap = len(query_terms.intersection(tokenize(cleaned)))
            candidates.append((overlap, cleaned))
    candidates.sort(key=lambda item: (-item[0], -len(item[1])))
    selected = [sentence for _, sentence in candidates[:2]]
    return " ".join(selected) if selected else "No supported answer was found."

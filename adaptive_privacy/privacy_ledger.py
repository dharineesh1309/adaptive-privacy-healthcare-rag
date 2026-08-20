"""Persistent, value-free disclosure accounting across browser sessions."""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import DocumentVariants, RiskLevel


QUASI_TYPES = {"age", "location", "date", "organization"}
SENSITIVE_TYPES = {"diagnosis", "salary", "balance", "treatment"}
PRINCIPAL_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{2,63}$")
LEVEL_ORDER = {RiskLevel.LOW: 0, RiskLevel.MEDIUM: 1, RiskLevel.HIGH: 2}


@dataclass(frozen=True)
class LedgerDecision:
    action: str
    minimum_tier: RiskLevel
    principal_ref: str
    predicted_new_exposures: list[dict[str, str]]
    reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "minimum_tier": self.minimum_tier.value,
            "principal_ref": self.principal_ref,
            "predicted_new_exposures": self.predicted_new_exposures,
            "reasons": self.reasons,
        }


class PersistentPrivacyLedger:
    def __init__(self, config: dict[str, Any], path: str | Path | None = None):
        self.config = config.get("ledger", {})
        self.path = Path(path) if path else None
        self.lock = threading.RLock()
        self.data: dict[str, Any] = {"version": 2, "principals": {}}
        self._load()

    @staticmethod
    def validate_principal_id(principal_id: str) -> str:
        candidate = principal_id.strip()
        if not PRINCIPAL_PATTERN.fullmatch(candidate):
            raise ValueError(
                "Principal ID must be 3-64 characters using letters, numbers, _ or -."
            )
        return candidate

    @classmethod
    def principal_hash(cls, principal_id: str) -> str:
        candidate = cls.validate_principal_id(principal_id)
        return hashlib.sha256(candidate.encode("utf-8")).hexdigest()

    def preview(
        self,
        principal_id: str,
        documents: dict[str, DocumentVariants],
        tiers: dict[str, RiskLevel],
        requested_types: set[str],
    ) -> LedgerDecision:
        with self.lock:
            principal_hash = self.principal_hash(principal_id)
            state = self._state(principal_hash)
            predicted = self._exposures(documents, tiers)
            existing = set(state["exposure_keys"])
            new = [item for item in predicted if self._key(item) not in existing]
            budgets = self._budgets(state)
            new_quasi = [item for item in new if item["category"] == "quasi"]
            new_sensitive = [item for item in new if item["category"] == "sensitive"]
            new_documents = {
                item["document_ref"] for item in new
            } - set(state["document_ids"])
            quasi_over = len(new_quasi) > budgets["remaining_quasi"]
            sensitive_over = len(new_sensitive) > budgets["remaining_sensitive"]
            documents_over = len(new_documents) > budgets["remaining_documents"]
            reasons = [
                f"Predicted marginal exposure: {len(new)} new attribute unit(s)"
            ]
            if not (quasi_over or sensitive_over or documents_over):
                reasons.append("Persistent principal budget permits this release")
                return LedgerDecision(
                    action="allow",
                    minimum_tier=RiskLevel.LOW,
                    principal_ref=principal_hash[:12],
                    predicted_new_exposures=new,
                    reasons=reasons,
                )

            exhausted_categories = set()
            if quasi_over:
                exhausted_categories.update(QUASI_TYPES)
                reasons.append("Principal quasi-identifier budget would be exceeded")
            if sensitive_over:
                exhausted_categories.update(SENSITIVE_TYPES)
                reasons.append("Principal sensitive-attribute budget would be exceeded")
            if documents_over:
                reasons.append("Principal protected-document budget would be exceeded")
            targeted = bool(requested_types & exhausted_categories)
            if targeted or documents_over:
                reasons.append("Requested disclosure is blocked across session boundaries")
                return LedgerDecision(
                    action="block",
                    minimum_tier=RiskLevel.HIGH,
                    principal_ref=principal_hash[:12],
                    predicted_new_exposures=new,
                    reasons=reasons,
                )
            minimum = RiskLevel.HIGH if sensitive_over else RiskLevel.MEDIUM
            reasons.append(f"Ledger raises minimum protection to {minimum.value}")
            return LedgerDecision(
                action="escalate",
                minimum_tier=minimum,
                principal_ref=principal_hash[:12],
                predicted_new_exposures=new,
                reasons=reasons,
            )

    def record_exposure(
        self,
        principal_id: str,
        purpose: str,
        documents: dict[str, DocumentVariants],
        tiers: dict[str, RiskLevel],
    ) -> dict[str, Any]:
        with self.lock:
            principal_hash = self.principal_hash(principal_id)
            state = self._state(principal_hash)
            exposures = self._exposures(documents, tiers)
            exposure_keys = set(state["exposure_keys"])
            for item in exposures:
                exposure_keys.add(self._key(item))
            state["exposure_keys"] = sorted(exposure_keys)
            exposed_documents = {item["document_ref"] for item in exposures}
            state["document_ids"] = sorted(
                set(state["document_ids"]) | exposed_documents
            )
            state["purposes"] = sorted(set(state["purposes"]) | {purpose})
            state["release_count"] += 1
            state["updated_at"] = time.time()
            self._save()
            return self.snapshot(principal_id)

    def snapshot(self, principal_id: str) -> dict[str, Any]:
        with self.lock:
            principal_hash = self.principal_hash(principal_id)
            state = self._state(principal_hash)
            budgets = self._budgets(state)
            exposures = [self._parse_key(item) for item in state["exposure_keys"]]
            return {
                "principal_ref": principal_hash[:12],
                "release_count": state["release_count"],
                "purposes": list(state["purposes"]),
                "document_count": len(state["document_ids"]),
                "exposed_types": sorted({item["entity_type"] for item in exposures}),
                "exposure_units": len(exposures),
                "remaining_budget": {
                    "quasi_attributes": budgets["remaining_quasi"],
                    "sensitive_attributes": budgets["remaining_sensitive"],
                    "documents": budgets["remaining_documents"],
                },
            }

    def reset(self, principal_id: str) -> dict[str, Any]:
        with self.lock:
            principal_hash = self.principal_hash(principal_id)
            self.data["principals"][principal_hash] = self._empty_state()
            self._save()
            return self.snapshot(principal_id)

    def _budgets(self, state: dict[str, Any]) -> dict[str, int]:
        exposures = [self._parse_key(item) for item in state["exposure_keys"]]
        quasi_used = sum(item["category"] == "quasi" for item in exposures)
        sensitive_used = sum(item["category"] == "sensitive" for item in exposures)
        limits = self.config.get("budgets", {})
        return {
            "remaining_quasi": max(
                0, int(limits.get("quasi_attributes", 3)) - quasi_used
            ),
            "remaining_sensitive": max(
                0, int(limits.get("sensitive_attributes", 2)) - sensitive_used
            ),
            "remaining_documents": max(
                0, int(limits.get("documents", 4)) - len(state["document_ids"])
            ),
        }

    @staticmethod
    def _exposures(
        documents: dict[str, DocumentVariants], tiers: dict[str, RiskLevel]
    ) -> list[dict[str, str]]:
        exposures: dict[str, dict[str, str]] = {}
        for document_id, tier in tiers.items():
            document = documents.get(document_id)
            if document is None:
                continue
            for entity in document.detected_entities:
                category = str(entity.get("category", "unknown"))
                entity_type = str(entity.get("entity_type", "unknown"))
                visible = (
                    tier is RiskLevel.LOW and category in {"quasi", "sensitive"}
                ) or (tier is RiskLevel.MEDIUM and category == "sensitive")
                if not visible:
                    continue
                item = {
                    "document_ref": PersistentPrivacyLedger._document_ref(document_id),
                    "entity_type": entity_type,
                    "category": category,
                }
                exposures[PersistentPrivacyLedger._key(item)] = item
        return [exposures[key] for key in sorted(exposures)]

    @staticmethod
    def _key(item: dict[str, str]) -> str:
        return f"{item['document_ref']}|{item['entity_type']}|{item['category']}"

    @staticmethod
    def _document_ref(document_id: str) -> str:
        return hashlib.sha256(document_id.encode("utf-8")).hexdigest()

    @staticmethod
    def _parse_key(value: str) -> dict[str, str]:
        document_ref, entity_type, category = value.split("|", 2)
        return {
            "document_ref": document_ref,
            "entity_type": entity_type,
            "category": category,
        }

    def _state(self, principal_hash: str) -> dict[str, Any]:
        return self.data["principals"].setdefault(principal_hash, self._empty_state())

    @staticmethod
    def _empty_state() -> dict[str, Any]:
        return {
            "exposure_keys": [],
            "document_ids": [],
            "purposes": [],
            "release_count": 0,
            "updated_at": time.time(),
        }

    def _load(self) -> None:
        if self.path is None or not self.path.exists():
            return
        loaded = json.loads(self.path.read_text(encoding="utf-8"))
        version = loaded.get("version")
        if version not in {1, 2} or not isinstance(loaded.get("principals"), dict):
            raise ValueError("Unsupported privacy-ledger format.")
        if version == 1:
            for state in loaded["principals"].values():
                migrated_keys = []
                for value in state.get("exposure_keys", []):
                    document_id, entity_type, category = value.split("|", 2)
                    document_ref = self._document_ref(document_id)
                    migrated_keys.append(
                        f"{document_ref}|{entity_type}|{category}"
                    )
                state["exposure_keys"] = sorted(set(migrated_keys))
                state["document_ids"] = sorted(
                    {self._document_ref(value) for value in state.get("document_ids", [])}
                )
            loaded["version"] = 2
        self.data = loaded
        if version == 1:
            self._save()

    def _save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(self.path)

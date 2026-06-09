"""Append-only JSONL audit logging for policy decisions."""

from __future__ import annotations

import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class AuditLogger:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None

    def write(self, event: dict[str, Any]) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        safe_event = dict(event)
        query = safe_event.pop("query", None)
        if query is not None:
            safe_event["query_sha256"] = hashlib.sha256(
                str(query).encode("utf-8")
            ).hexdigest()
        # These fields are deliberately never persisted by the audit logger.
        safe_event.pop("protected_contexts", None)
        safe_event.pop("answer", None)
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **safe_event,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

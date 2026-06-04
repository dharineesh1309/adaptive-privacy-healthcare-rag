"""JSON input/output helpers for documents and tiered representations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import DocumentVariants


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: str | Path, value: Any) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def load_tiered_documents(path: str | Path) -> list[DocumentVariants]:
    payload = load_json(path)
    if not isinstance(payload, list):
        raise ValueError("Tiered document file must contain a JSON array.")
    return [DocumentVariants.from_dict(item) for item in payload]


def save_tiered_documents(
    path: str | Path, documents: list[DocumentVariants]
) -> None:
    save_json(path, [document.to_dict() for document in documents])

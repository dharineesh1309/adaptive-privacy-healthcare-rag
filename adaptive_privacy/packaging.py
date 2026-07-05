"""Build a clean, auditable lab-submission archive."""

from __future__ import annotations

import hashlib
import json
import zipfile
from datetime import date
from pathlib import Path
from typing import Iterable


SOURCE_DIRECTORIES = (
    "adaptive_privacy",
    "agents",
    "assets",
    "configs",
    "data_preparation",
    "docs",
    "examples",
    "prompts",
    "retrieval",
    "scripts",
    "tests",
    "web_demo",
)

SOURCE_FILES = (
    "README.md",
    "requirements.txt",
    "run_pipeline.py",
)

RESULT_FILES = {
    "results/adaptive_privacy/final_lab_report.md": "FINAL_LAB_REPORT.md",
    "results/adaptive_privacy/heldout/summary.json": "experiment_results/controlled_summary.json",
    "results/adaptive_privacy/heldout/evaluation.csv": "experiment_results/controlled_evaluation.csv",
    "results/adaptive_privacy/chatdoctor/summary.json": "experiment_results/chatdoctor_summary.json",
    "results/adaptive_privacy/chatdoctor/evaluation.csv": "experiment_results/chatdoctor_evaluation.csv",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def iter_source_files(project_root: Path) -> Iterable[tuple[Path, str]]:
    for filename in SOURCE_FILES:
        path = project_root / filename
        if path.is_file():
            yield path, filename
    for directory in SOURCE_DIRECTORIES:
        root = project_root / directory
        if not root.exists():
            continue
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
                continue
            yield path, path.relative_to(project_root).as_posix()


def create_submission_archive(project_root: Path, output_path: Path) -> dict:
    missing_results = [
        project_root / source
        for source in RESULT_FILES
        if not (project_root / source).is_file()
    ]
    if missing_results:
        missing = ", ".join(str(path) for path in missing_results)
        raise FileNotFoundError(
            f"Required result files are missing: {missing}. Run the evaluation and "
            "report scripts before packaging."
        )

    entries = list(iter_source_files(project_root))
    for source, archive_name in RESULT_FILES.items():
        entries.append((project_root / source, archive_name))
    entries.sort(key=lambda item: item[1])

    file_manifest = [
        {
            "path": archive_name,
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path, archive_name in entries
    ]
    chatdoctor_summary = json.loads(
        (project_root / "results/adaptive_privacy/chatdoctor/summary.json").read_text(
            encoding="utf-8"
        )
    )
    manifest = {
        "project": "Role- and Query-Risk-Aware Adaptive Privacy for RAG",
        "packaged_on": date.today().isoformat(),
        "files": len(file_manifest),
        "source_data_included": False,
        "model_weights_included": False,
        "chatdoctor_source": {
            "dataset": chatdoctor_summary["benchmark"].get("source"),
            "revision": chatdoctor_summary["benchmark"].get("source_revision"),
            "rows_used": chatdoctor_summary["benchmark"].get("source_rows"),
            "note": "Source rows are excluded; rerun the documented fetch script.",
        },
        "license_note": (
            "The upstream repository contains no LICENSE file. Treat this package "
            "as a coursework artifact unless redistribution permission is clarified."
        ),
        "contents": file_manifest,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        output_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for path, archive_name in entries:
            archive.write(path, archive_name)
        archive.writestr(
            "SUBMISSION_MANIFEST.json",
            json.dumps(manifest, ensure_ascii=False, indent=2),
        )
    return manifest

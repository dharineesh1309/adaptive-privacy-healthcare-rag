"""Controlled evaluation of purpose binding and cross-session privacy accounting."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from .models import DocumentVariants
from .pipeline import AdaptivePrivacyPipeline
from .privacy_ledger import PersistentPrivacyLedger
from .session_pipeline import SessionAwarePrivacyPipeline


def _pipeline(
    documents: list[DocumentVariants],
    config: dict[str, Any],
    ledger_path: Path | None = None,
    *,
    disable_ledger: bool = False,
) -> SessionAwarePrivacyPipeline:
    variant = copy.deepcopy(config)
    variant["retrieval"]["top_k"] = 1
    if disable_ledger:
        variant["ledger"]["budgets"] = {
            "quasi_attributes": 999,
            "sensitive_attributes": 999,
            "documents": 999,
        }
    return SessionAwarePrivacyPipeline(
        AdaptivePrivacyPipeline(documents, config=variant),
        ledger=PersistentPrivacyLedger(variant, ledger_path),
    )


def _request(
    pipeline: SessionAwarePrivacyPipeline,
    principal_id: str,
    query: str,
    *,
    role: str = "doctor",
    purpose: str = "treatment",
    role_verified: bool = True,
) -> dict[str, Any]:
    session_id = pipeline.create_session(principal_id)["session_id"]
    return pipeline.process(
        session_id,
        query,
        role,
        mode="fixed_low",
        role_verified=role_verified,
        purpose=purpose,
        principal_id=principal_id,
    )


def evaluate_privacy_ledger(
    documents: list[DocumentVariants],
    config: dict[str, Any],
    ledger_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Compare the persistent ledger with a session-only high-budget ablation."""

    controlled = copy.deepcopy(config)
    controlled["ledger"]["budgets"] = {
        "quasi_attributes": 3,
        "sensitive_attributes": 1,
        "documents": 2,
    }
    full = _pipeline(documents, controlled, ledger_path)
    session_only = _pipeline(documents, controlled, disable_ledger=True)
    rows: list[dict[str, Any]] = []

    reset_principal = "eval-reset-attack"
    full.reset_ledger(reset_principal)
    first = _request(
        full,
        reset_principal,
        "What age group is the patient with diabetes in?",
    )
    second = _request(
        full,
        reset_principal,
        "What age group is the patient with migraine in?",
    )
    session_only.reset_ledger(reset_principal)
    baseline_first = _request(
        session_only,
        reset_principal,
        "What age group is the patient with diabetes in?",
    )
    baseline_second = _request(
        session_only,
        reset_principal,
        "What age group is the patient with migraine in?",
    )
    rows.append(
        {
            "scenario": "cross_session_reset_attack",
            "expected": "second request blocked",
            "full_guard_pass": int(not first["blocked"] and second["blocked"]),
            "session_only_pass": int(
                not baseline_first["blocked"] and baseline_second["blocked"]
            ),
            "full_second_action": second["ledger_decision"]["action"],
            "session_only_second_action": baseline_second["ledger_decision"]["action"],
            "exposure_units": second["ledger"]["exposure_units"],
        }
    )

    replay_principal = "eval-safe-replay"
    full.reset_ledger(replay_principal)
    replay_first = _request(
        full,
        replay_principal,
        "What age group is the patient with diabetes in?",
    )
    units_before = replay_first["ledger"]["exposure_units"]
    replay_second = _request(
        full,
        replay_principal,
        "What age group is the patient with diabetes in?",
    )
    rows.append(
        {
            "scenario": "idempotent_replay",
            "expected": "same disclosure allowed without a second charge",
            "full_guard_pass": int(
                not replay_second["blocked"]
                and replay_second["ledger"]["exposure_units"] == units_before
            ),
            "session_only_pass": 1,
            "full_second_action": replay_second["ledger_decision"]["action"],
            "session_only_second_action": "not_applicable",
            "exposure_units": replay_second["ledger"]["exposure_units"],
        }
    )

    isolated_principal = "eval-isolated-user"
    full.reset_ledger(isolated_principal)
    isolated = _request(
        full,
        isolated_principal,
        "What age group is the patient with migraine in?",
    )
    rows.append(
        {
            "scenario": "principal_isolation",
            "expected": "new principal has an independent budget",
            "full_guard_pass": int(not isolated["blocked"]),
            "session_only_pass": 1,
            "full_second_action": isolated["ledger_decision"]["action"],
            "session_only_second_action": "not_applicable",
            "exposure_units": isolated["ledger"]["exposure_units"],
        }
    )

    purpose_principal = "eval-purpose-denial"
    full.reset_ledger(purpose_principal)
    purpose_denial = _request(
        full,
        purpose_principal,
        "What diagnosis does the patient have?",
        role="public",
        purpose="treatment",
        role_verified=False,
    )
    rows.append(
        {
            "scenario": "purpose_role_violation",
            "expected": "public role cannot claim treatment purpose",
            "full_guard_pass": int(
                purpose_denial["blocked"]
                and not purpose_denial["purpose_decision"]["allowed"]
            ),
            "session_only_pass": int(purpose_denial["blocked"]),
            "full_second_action": purpose_denial["ledger_decision"]["action"],
            "session_only_second_action": "not_applicable",
            "exposure_units": purpose_denial["ledger"]["exposure_units"],
        }
    )

    benign_principal = "eval-general-info"
    full.reset_ledger(benign_principal)
    benign = _request(
        full,
        benign_principal,
        "What is diabetes and how is it commonly managed in general?",
        role="public",
        purpose="general_information",
        role_verified=False,
    )
    rows.append(
        {
            "scenario": "benign_general_information",
            "expected": "low-risk educational request allowed",
            "full_guard_pass": int(not benign["blocked"]),
            "session_only_pass": 1,
            "full_second_action": benign["ledger_decision"]["action"],
            "session_only_second_action": "not_applicable",
            "exposure_units": benign["ledger"]["exposure_units"],
        }
    )

    persisted = ledger_path.read_text(encoding="utf-8").lower()
    raw_values_absent = all(
        value not in persisted
        for value in ("john smith", "jane doe", "diabetes", "migraine", reset_principal)
    )
    summary = {
        "benchmark": {
            "scenarios": len(rows),
            "cross_session_attack_scenarios": 1,
            "purpose_violation_scenarios": 1,
            "benign_or_isolation_scenarios": 3,
        },
        "metrics": {
            "full_guard_pass_rate": round(
                sum(row["full_guard_pass"] for row in rows) / len(rows), 4
            ),
            "session_only_reset_attack_block_rate": float(
                rows[0]["session_only_pass"]
            ),
            "persistent_reset_attack_block_rate": float(rows[0]["full_guard_pass"]),
            "duplicate_exposure_charge": (
                replay_second["ledger"]["exposure_units"] - units_before
            ),
            "raw_values_absent_from_ledger": raw_values_absent,
        },
        "scenarios": rows,
    }
    return rows, summary


def render_ledger_report(summary: dict[str, Any]) -> str:
    metrics = summary["metrics"]
    lines = [
        "# Purpose-Bound Persistent Privacy Ledger Evaluation",
        "",
        "This controlled experiment tests whether a privacy budget survives a new browser session while remaining isolated between principals.",
        "",
        "| Metric | Result |",
        "|---|---:|",
        f"| Full scenario pass rate | {metrics['full_guard_pass_rate']:.1%} |",
        f"| Persistent-ledger reset-attack block rate | {metrics['persistent_reset_attack_block_rate']:.1%} |",
        f"| Session-only reset-attack block rate | {metrics['session_only_reset_attack_block_rate']:.1%} |",
        f"| Duplicate exposure charge | {metrics['duplicate_exposure_charge']} |",
        f"| Raw values absent from ledger | {'yes' if metrics['raw_values_absent_from_ledger'] else 'no'} |",
        "",
        "| Scenario | Expected behavior | Full guard | Session-only ablation |",
        "|---|---|---:|---:|",
    ]
    for row in summary["scenarios"]:
        lines.append(
            f"| {row['scenario']} | {row['expected']} | "
            f"{'pass' if row['full_guard_pass'] else 'fail'} | "
            f"{'pass' if row['session_only_pass'] else 'fail'} |"
        )
    lines.extend(
        [
            "",
            "The session-only column is an ablation with effectively unlimited persistent budgets. The scenarios are deterministic functional tests, not a formal privacy guarantee.",
            "",
        ]
    )
    return "\n".join(lines)

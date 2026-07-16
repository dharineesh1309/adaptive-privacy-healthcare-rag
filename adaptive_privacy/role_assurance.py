"""Deterministic role-claim resolution for disclosure policy decisions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


PRIVILEGED_ROLES = {"doctor", "admin"}


@dataclass(frozen=True)
class RoleAssuranceDecision:
    requested_role: str
    effective_role: str
    verified: bool
    privileged: bool
    downgraded: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RoleAssurancePolicy:
    """Treat privileged roles as public unless an external verifier confirms them.

    The demo accepts a boolean verification flag only to simulate the output of an
    authentication/authorization layer. Production code must derive this value
    from trusted identity middleware, never from a user-controlled request body.
    """

    def resolve(self, requested_role: str, verified: bool = False) -> RoleAssuranceDecision:
        requested = requested_role.strip().lower() or "unknown"
        privileged = requested in PRIVILEGED_ROLES
        is_verified = bool(verified and privileged)
        downgraded = privileged and not is_verified
        effective = "public" if downgraded else requested
        if downgraded:
            reason = "Privileged role was not verified; public-role policy applied"
        elif is_verified:
            reason = "Privileged role was verified by the simulated assurance layer"
        else:
            reason = "Role does not require privileged-role verification"
        return RoleAssuranceDecision(
            requested_role=requested,
            effective_role=effective,
            verified=is_verified,
            privileged=privileged,
            downgraded=downgraded,
            reason=reason,
        )

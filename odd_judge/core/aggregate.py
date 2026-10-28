"""Aggregate per-rule results into a JudgmentReport."""

from __future__ import annotations

from typing import Any

from .types import (
    JudgmentReport,
    RuleResult,
    ScenarioState,
    Severity,
)

_STATE_ORDER = {
    ScenarioState.OUTSIDE: 3,
    ScenarioState.BOUNDARY: 2,
    ScenarioState.INSIDE: 1,
    ScenarioState.UNDEFINED: 0,
    ScenarioState.SKIPPED: -1,
}

_RECOMMENDATION = {
    ScenarioState.INSIDE: "continue",
    ScenarioState.BOUNDARY: "warn_driver",
    ScenarioState.OUTSIDE: "mrm_or_refuse",
    ScenarioState.UNDEFINED: "continue",
    ScenarioState.SKIPPED: "continue",
}


def aggregate(
    profile_id: str,
    results: list[RuleResult],
    skipped: list[dict[str, Any]] | None = None,
    undefined_handled: list[dict[str, Any]] | None = None,
    active_switches: list[str] | None = None,
    timestamp: str = "",
    definition_mode: str = "",
    boundary_warnings_enabled: bool = True,
) -> JudgmentReport:
    """Combine rule results into an overall judgment."""
    skipped = skipped or []
    undefined_handled = undefined_handled or []
    active_switches = active_switches or []

    # Classify
    outside_reasons: list[str] = []
    boundary_reasons: list[str] = []
    contributing = [r for r in results if r.state not in (ScenarioState.SKIPPED, ScenarioState.UNDEFINED)]

    for r in results:
        if r.state == ScenarioState.OUTSIDE:
            code = r.reason_code or r.element_id
            outside_reasons.append(f"{code}: {r.message_zh}")
        elif r.state == ScenarioState.BOUNDARY and boundary_warnings_enabled:
            code = r.reason_code or r.element_id
            boundary_reasons.append(f"{code}: {r.message_zh}")

    # Overall state: worst of contributing
    if contributing:
        worst = max(contributing, key=lambda r: _STATE_ORDER[r.state])
        state = worst.state
    else:
        state = ScenarioState.UNDEFINED

    # If boundary warnings disabled, demote BOUNDARY to INSIDE for overall
    if state == ScenarioState.BOUNDARY and not boundary_warnings_enabled:
        state = ScenarioState.INSIDE

    # Severity
    if state == ScenarioState.OUTSIDE:
        severity = Severity.CRITICAL
    elif state == ScenarioState.BOUNDARY:
        severity = Severity.WARNING
    else:
        severity = Severity.INFO

    recommendation = _RECOMMENDATION.get(state, "continue")

    return JudgmentReport(
        profile_id=profile_id,
        state=state,
        severity=severity,
        results=results,
        outside_reasons=outside_reasons,
        boundary_reasons=boundary_reasons,
        undefined_handled=undefined_handled,
        skipped=skipped,
        recommendation=recommendation,
        active_switches=active_switches,
        timestamp=timestamp,
        definition_mode=definition_mode,
    )

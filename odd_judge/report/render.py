"""Render JudgmentReport as JSON or text table."""

from __future__ import annotations

from ..core.types import JudgmentReport, ScenarioState


def to_json(report: JudgmentReport, indent: int = 2) -> str:
    return report.to_json(indent=indent)


def to_text(report: JudgmentReport) -> str:
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append(f"ODD Judgment Report  |  profile={report.profile_id}  mode={report.definition_mode}")
    lines.append(f"Overall: {report.state.value}  severity={report.severity.value}  → {report.recommendation}")
    lines.append("=" * 72)

    if report.outside_reasons:
        lines.append("\n[OUTSIDE / 越界]")
        for r in report.outside_reasons:
            lines.append(f"  ✗ {r}")

    if report.boundary_reasons:
        lines.append("\n[BOUNDARY / 临近边界]")
        for r in report.boundary_reasons:
            lines.append(f"  ⚠ {r}")

    # Compact results table
    lines.append("\n[Rule Results]")
    lines.append(f"  {'element_id':<36} {'state':<10} {'observed':<12} {'margin':<10} {'code'}")
    lines.append("  " + "-" * 68)
    for r in report.results:
        obs = f"{r.observed}" if r.observed is not None else "-"
        margin = f"{r.remaining_margin:.2f}" if r.remaining_margin is not None else "-"
        code = r.reason_code or ""
        lines.append(
            f"  {r.element_id:<36} {r.state.value:<10} {obs:<12} {margin:<10} {code}"
        )

    if report.skipped:
        lines.append(f"\n[Skipped by switch] {len(report.skipped)} rule(s)")
    if report.undefined_handled:
        lines.append(f"[Undefined handled] {len(report.undefined_handled)} rule(s)")

    lines.append(f"\nTimestamp: {report.timestamp}")
    return "\n".join(lines)


def render(report: JudgmentReport, fmt: str = "json") -> str:
    if fmt == "text":
        return to_text(report)
    return to_json(report)

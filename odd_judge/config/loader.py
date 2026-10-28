"""Load and merge JSON profiles, switches, and observation files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..core.composite import parse_composite
from ..core.errors import ConfigError
from ..core.types import (
    BoundaryMargin,
    DefinitionMode,
    Requirement,
    Rule,
    ThresholdSpec,
)
from .registry import get_extra_elements
from .schema import validate_profile_dict, validate_switches_dict


def _load_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"File not found: {p}")
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ConfigError(f"Invalid JSON in {p}: {e}") from e


def _parse_margin(data: dict[str, Any] | None) -> BoundaryMargin | None:
    if not data:
        return None
    return BoundaryMargin(mode=data.get("mode", "absolute"), value=float(data.get("value", 0.0)))


def _parse_spec(el: dict[str, Any]) -> ThresholdSpec | None:
    rt = el.get("rule_type")
    if rt != "numeric_range":
        return None
    margin = _parse_margin(el.get("boundary_margin"))
    rng = el.get("range")
    return ThresholdSpec(
        op=el.get("op", ">="),
        threshold=el.get("threshold"),
        range=tuple(rng) if rng else None,
        values=tuple(el["values"]) if el.get("values") else None,
        expect=el.get("expect"),
        margin=margin,
    )


def _parse_rule(el: dict[str, Any]) -> Rule:
    req = Requirement(el.get("requirement", "allowed"))
    undef_as = el.get("undefined_as")
    applies = el.get("applies_when")
    return Rule(
        element_id=el["element_id"],
        path=tuple(el.get("path", [])),
        requirement=req,
        rule_type=el.get("rule_type", "qualitative_bool"),
        observation_key=el.get("observation_key", ""),
        label_zh=el.get("label_zh", ""),
        qualifier=el.get("qualifier", "include"),
        spec=_parse_spec(el),
        allowed_values=tuple(el["allowed_values"]) if el.get("allowed_values") else None,
        forbidden_values=tuple(el["forbidden_values"]) if el.get("forbidden_values") else None,
        boundary_values=tuple(el["boundary_values"]) if el.get("boundary_values") else None,
        expect=el.get("expect"),
        applies_when=parse_composite(applies) if applies else None,
        undefined_as=Requirement(undef_as) if undef_as else None,
        warning_codes=dict(el.get("warning_codes", {})),
        tags=tuple(el.get("tags", [])),
        relation=el.get("relation"),
        road_rule_match=el.get("road_rule_match"),
        confidence_threshold=el.get("confidence_threshold"),
        confidence_band=el.get("confidence_band"),
    )


def parse_profile(data: dict[str, Any]) -> tuple[str, DefinitionMode, list[Rule]]:
    errors = validate_profile_dict(data)
    if errors:
        raise ConfigError("Profile validation failed:\n  " + "\n  ".join(errors))

    profile_id = data["profile_id"]
    mode = DefinitionMode(data.get("definition_mode", "default"))
    rules = [_parse_rule(el) for el in data["elements"]]

    # Append registry-registered extra elements
    for el in get_extra_elements().values():
        rules.append(_parse_rule(el))

    return profile_id, mode, rules


def load_profile(path: str | Path) -> tuple[str, DefinitionMode, list[Rule]]:
    data = _load_json(path)
    return parse_profile(data)


def merge_profiles(
    base: dict[str, Any],
    *others: dict[str, Any],
    base_priority: bool = True,
) -> dict[str, Any]:
    """Merge profile dicts. On element_id collision, base wins if base_priority."""
    merged_elements: dict[str, dict[str, Any]] = {}
    order: list[str] = []

    sources = [base, *others] if base_priority else [*others, base]
    for src in sources:
        for el in src.get("elements", []):
            eid = el.get("element_id")
            if not eid:
                continue
            if eid not in merged_elements:
                order.append(eid)
            # later sources do not overwrite earlier if base_priority
            # with base_priority=True, base is first so it wins
            if eid not in merged_elements:
                merged_elements[eid] = el
            elif not base_priority:
                merged_elements[eid] = el

    result = dict(base)
    result["elements"] = [merged_elements[eid] for eid in order]
    return result


def load_profile_with_extends(
    path: str | Path,
    extends_paths: list[str | Path] | None = None,
    override_paths: list[str | Path] | None = None,
) -> tuple[str, DefinitionMode, list[Rule]]:
    base = _load_json(path)
    others = [_load_json(p) for p in (extends_paths or [])]
    if others:
        base = merge_profiles(base, *others, base_priority=True)

    # Overrides: same element_id replaces; new element_id appends
    for op in (override_paths or []):
        od = _load_json(op)
        by_id = {el["element_id"]: el for el in base.get("elements", []) if "element_id" in el}
        for el in od.get("elements", []):
            eid = el.get("element_id")
            if eid and eid in by_id:
                by_id[eid] = el
            elif eid:
                by_id[eid] = el
        base["elements"] = list(by_id.values())

    return parse_profile(base)


def load_switches(path: str | Path | None = None) -> dict[str, Any]:
    if path is None:
        return {
            "global_enabled": True,
            "by_element": {},
            "by_rule_type": {},
            "by_tag": {},
            "boundary_warnings_enabled": True,
            "boundary_severity": "warning",
        }
    data = _load_json(path)
    errors = validate_switches_dict(data)
    if errors:
        raise ConfigError("Switches validation failed:\n  " + "\n  ".join(errors))
    defaults = {
        "global_enabled": True,
        "by_element": {},
        "by_rule_type": {},
        "by_tag": {},
        "boundary_warnings_enabled": True,
        "boundary_severity": "warning",
    }
    defaults.update(data)
    return defaults


def load_observation(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"Observation file not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

"""Load and merge JSON profiles, switches, and observation files."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..core.composite import parse_composite
from ..core.errors import ConfigError
from ..core.types import (
    BoundaryMargin,
    DefinitionMode,
    MissingPolicy,
    Requirement,
    Rule,
    ThresholdSpec,
)
from .registry import get_extra_elements
from .schema import (
    check_warning_code_duplicates,
    validate_profile_dict,
    validate_switches_dict,
)


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


def parse_missing_policy(data: dict[str, Any]) -> MissingPolicy | None:
    """Build a ``MissingPolicy`` from a profile dict, or ``None`` when absent."""
    raw = data.get("missing_policy")
    if raw is None:
        return None
    return MissingPolicy(
        by_element=dict(raw.get("by_element") or {}),
        by_tag=dict(raw.get("by_tag") or {}),
        default=str(raw.get("default", "mode")),
    )


@dataclass(frozen=True)
class ProfileBundle:
    """A parsed profile plus its missing-observation policy."""

    profile_id: str
    definition_mode: DefinitionMode
    rules: list[Rule]
    missing_policy: MissingPolicy | None = None


def _parse_profile_core(
    data: dict[str, Any],
) -> tuple[str, DefinitionMode, list[Rule], MissingPolicy | None]:
    errors = validate_profile_dict(data)
    if errors:
        raise ConfigError("Profile validation failed:\n  " + "\n  ".join(errors))

    profile_id = data["profile_id"]
    mode = DefinitionMode(data.get("definition_mode", "default"))
    rules = [_parse_rule(el) for el in data["elements"]]

    # Append registry-registered extra elements (cross-source duplicate check)
    extras = get_extra_elements()
    dup_errors = check_warning_code_duplicates(list(data["elements"]) + list(extras.values()))
    if dup_errors:
        raise ConfigError("Profile validation failed:\n  " + "\n  ".join(dup_errors))
    for el in extras.values():
        rules.append(_parse_rule(el))

    return profile_id, mode, rules, parse_missing_policy(data)


def parse_profile(data: dict[str, Any]) -> tuple[str, DefinitionMode, list[Rule]]:
    profile_id, mode, rules, _ = _parse_profile_core(data)
    return profile_id, mode, rules


def parse_profile_bundle(data: dict[str, Any]) -> ProfileBundle:
    profile_id, mode, rules, policy = _parse_profile_core(data)
    return ProfileBundle(profile_id, mode, rules, policy)


def load_profile(path: str | Path) -> tuple[str, DefinitionMode, list[Rule]]:
    data = _load_json(path)
    return parse_profile(data)


def _merge_missing_policy(*policies: dict[str, Any] | None) -> dict[str, Any] | None:
    """Overlay missing_policy blocks; later sources win per sub-key.

    Elements merge with ``base_priority`` (base wins), but policy refines safety
    semantics per source, so the later (more specific) source wins.
    """
    merged: dict[str, Any] = {}
    seen = False
    for pol in policies:
        if not pol:
            continue
        seen = True
        for group in ("by_element", "by_tag"):
            sub = pol.get(group)
            if isinstance(sub, dict):
                merged.setdefault(group, {}).update(sub)
        if "default" in pol:
            merged["default"] = pol["default"]
    return merged if seen else None


def _apply_overrides(
    base: dict[str, Any],
    override_paths: list[str | Path] | None,
) -> dict[str, Any]:
    """Same element_id replaces; new element_id appends; policy overlays later-wins."""
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
        base["missing_policy"] = _merge_missing_policy(
            base.get("missing_policy"), od.get("missing_policy")
        )
    return base


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
    result["missing_policy"] = _merge_missing_policy(
        *[src.get("missing_policy") for src in sources]
    )
    return result


def load_profile_with_extends(
    path: str | Path,
    extends_paths: list[str | Path] | None = None,
    override_paths: list[str | Path] | None = None,
) -> tuple[str, DefinitionMode, list[Rule]]:
    bundle = load_profile_bundle(path, extends_paths, override_paths)
    return bundle.profile_id, bundle.definition_mode, bundle.rules


def load_profile_bundle(
    path: str | Path,
    extends_paths: list[str | Path] | None = None,
    override_paths: list[str | Path] | None = None,
) -> ProfileBundle:
    base = _load_json(path)
    others = [_load_json(p) for p in (extends_paths or [])]
    if others:
        base = merge_profiles(base, *others, base_priority=True)
    base = _apply_overrides(base, override_paths)
    return parse_profile_bundle(base)


def load_switches(path: str | Path | None = None) -> dict[str, Any]:
    if path is None:
        return {
            "global_enabled": True,
            "by_element": {},
            "by_rule_type": {},
            "by_tag": {},
            "boundary_warnings_enabled": True,
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
    }
    defaults.update(data)
    return defaults


def load_observation(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"Observation file not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

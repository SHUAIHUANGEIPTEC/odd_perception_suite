"""Lightweight structural validation for profiles and switches."""

from __future__ import annotations

from typing import Any

from ..core.errors import ConfigError

VALID_REQUIREMENTS = {"allowed", "not_allowed", "undefined"}
VALID_RULE_TYPES = {"numeric_range", "qualitative_enum", "qualitative_bool"}
VALID_OPS = {">=", "<=", ">", "<", "between", "in", "not_in", "==", "!=", "bool"}
VALID_MARGIN_MODES = {"absolute", "relative"}
VALID_DEFINITION_MODES = {"permissive", "restrictive", "default"}


def validate_profile_dict(data: dict[str, Any]) -> list[str]:
    """Return a list of error strings (empty = valid)."""
    errors: list[str] = []
    if not isinstance(data, dict):
        return ["profile must be a JSON object"]

    if "profile_id" not in data:
        errors.append("missing profile_id")
    if "elements" not in data:
        errors.append("missing elements")
        return errors

    mode = data.get("definition_mode", "default")
    if mode not in VALID_DEFINITION_MODES:
        errors.append(f"invalid definition_mode: {mode!r}")

    elements = data["elements"]
    if not isinstance(elements, list):
        errors.append("elements must be a list")
        return errors

    seen_ids: set[str] = set()
    for i, el in enumerate(elements):
        prefix = f"elements[{i}]"
        if not isinstance(el, dict):
            errors.append(f"{prefix}: must be object")
            continue
        eid = el.get("element_id")
        if not eid:
            errors.append(f"{prefix}: missing element_id")
        elif eid in seen_ids:
            errors.append(f"{prefix}: duplicate element_id {eid}")
        else:
            seen_ids.add(eid)

        req = el.get("requirement", "allowed")
        if req not in VALID_REQUIREMENTS:
            errors.append(f"{prefix} ({eid}): invalid requirement {req!r}")

        rt = el.get("rule_type")
        if rt is not None and rt not in VALID_RULE_TYPES:
            # allow registered custom types — only warn-level, not error
            # but validate the known ones strictly
            if rt not in ("numeric_range", "qualitative_enum", "qualitative_bool"):
                pass  # custom type, skip strict check

        if not el.get("observation_key") and req != "undefined":
            errors.append(f"{prefix} ({eid}): missing observation_key")

        if rt == "numeric_range":
            op = el.get("op")
            if op not in VALID_OPS:
                errors.append(f"{prefix} ({eid}): invalid op {op!r}")
            if op in (">=", "<=", ">", "<") and el.get("threshold") is None:
                errors.append(f"{prefix} ({eid}): op {op} requires threshold")
            if op == "between" and el.get("range") is None:
                errors.append(f"{prefix} ({eid}): op between requires range")

        if rt == "qualitative_enum" and req != "undefined" and not el.get("allowed_values") and not el.get("forbidden_values"):
            errors.append(f"{prefix} ({eid}): enum rule needs allowed_values or forbidden_values")

        bm = el.get("boundary_margin")
        if bm is not None:
            if not isinstance(bm, dict):
                errors.append(f"{prefix} ({eid}): boundary_margin must be object")
            else:
                if bm.get("mode") not in VALID_MARGIN_MODES:
                    errors.append(f"{prefix} ({eid}): invalid boundary_margin.mode")

    return errors


def validate_switches_dict(data: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not isinstance(data, dict):
        return ["switches must be a JSON object"]
    for key in ("by_element", "by_rule_type", "by_tag"):
        if key in data and not isinstance(data[key], dict):
            errors.append(f"{key} must be an object")
    return errors

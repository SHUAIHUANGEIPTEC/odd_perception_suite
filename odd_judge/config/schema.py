"""Lightweight structural validation for profiles and switches."""

from __future__ import annotations

import re
from typing import Any

from ..core.errors import ConfigError

VALID_REQUIREMENTS = {"allowed", "not_allowed", "undefined"}
VALID_RULE_TYPES = {"numeric_range", "qualitative_enum", "qualitative_bool"}
VALID_OPS = {">=", "<=", ">", "<", "between", "in", "not_in", "==", "!=", "bool"}
VALID_MARGIN_MODES = {"absolute", "relative"}
VALID_DEFINITION_MODES = {"permissive", "restrictive", "default"}
VALID_MISSING_POLICY_VALUES = {"allowed", "not_allowed", "undefined", "mode"}
VALID_WARNING_CODE_KEYS = {"outside", "boundary"}
WARNING_CODE_RE = re.compile(r"^[A-Z][A-Z0-9]*(-[A-Z0-9]+)+$")


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

    errors.extend(_check_missing_policy(data.get("missing_policy")))

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

        wc = el.get("warning_codes")
        if wc is not None:
            if not isinstance(wc, dict):
                errors.append(f"{prefix} ({eid}): warning_codes must be object")
            else:
                for key, val in wc.items():
                    if key not in VALID_WARNING_CODE_KEYS:
                        errors.append(
                            f"{prefix} ({eid}): invalid warning_codes key {key!r} "
                            f"(allowed: outside, boundary)"
                        )
                    if not isinstance(val, str) or not WARNING_CODE_RE.match(val):
                        errors.append(f"{prefix} ({eid}): invalid warning code format: {val!r}")

    errors.extend(check_warning_code_duplicates(elements))
    return errors


def _check_missing_policy(mp: object) -> list[str]:
    """Validate the optional top-level ``missing_policy`` block."""
    if mp is None:
        return []
    if not isinstance(mp, dict):
        return ["missing_policy must be an object"]
    errors: list[str] = []
    for key in mp:
        if key not in ("by_element", "by_tag", "default"):
            errors.append(f"missing_policy: unknown key {key!r}")
    for group in ("by_element", "by_tag"):
        sub = mp.get(group)
        if sub is None:
            continue
        if not isinstance(sub, dict):
            errors.append(f"missing_policy.{group} must be an object")
            continue
        for name, value in sub.items():
            if value not in VALID_MISSING_POLICY_VALUES:
                errors.append(
                    f"missing_policy.{group}[{name}]: invalid value {value!r} "
                    f"(allowed: {sorted(VALID_MISSING_POLICY_VALUES)})"
                )
    default = mp.get("default")
    if default is not None and default not in VALID_MISSING_POLICY_VALUES:
        errors.append(
            f"missing_policy.default: invalid value {default!r} "
            f"(allowed: {sorted(VALID_MISSING_POLICY_VALUES)})"
        )
    return errors


def check_warning_code_duplicates(elements: list[Any]) -> list[str]:
    """Return errors when the same warning code is used by more than one element.

    `boundary` and `outside` codes are counted uniformly; a code may appear
    at most once across the whole profile (duplicate codes make report
    consumers unable to attribute a warning to its source element).
    """
    errors: list[str] = []
    code_owners: dict[str, str] = {}
    for i, el in enumerate(elements):
        if not isinstance(el, dict):
            continue
        eid = el.get("element_id") or f"elements[{i}]"
        wc = el.get("warning_codes")
        if not isinstance(wc, dict):
            continue
        for val in wc.values():
            if not isinstance(val, str):
                continue
            owner = code_owners.get(val)
            if owner is None:
                code_owners[val] = eid
            elif owner != eid:
                errors.append(f"duplicate warning code {val!r}: used by {owner} and {eid}")
    return errors


def validate_switches_dict(data: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not isinstance(data, dict):
        return ["switches must be a JSON object"]
    for key in ("by_element", "by_rule_type", "by_tag"):
        if key in data and not isinstance(data[key], dict):
            errors.append(f"{key} must be an object")
    return errors

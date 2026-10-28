"""Rule type and element registry for extensibility."""

from __future__ import annotations

from typing import Any, Callable

from ..core.types import Rule

# rule_type -> evaluator function name (looked up in evaluate module by default)
_RULE_TYPES: dict[str, str] = {}
# element_id -> Rule (for plugin-registered elements)
_EXTRA_ELEMENTS: dict[str, dict[str, Any]] = {}


def register_rule_type(name: str) -> Callable:
    """Decorator to register a custom rule type name.

    The actual evaluator is resolved by name from odd_judge.core.evaluate
    or from a custom evaluator map.
    """

    def deco(fn: Callable) -> Callable:
        _RULE_TYPES[name] = getattr(fn, "__name__", name)
        return fn

    return deco


def register_element(element_dict: dict[str, Any]) -> None:
    """Register an extra ODC element (dict form matching profile schema)."""
    eid = element_dict.get("element_id")
    if not eid:
        raise ValueError("element_dict must have element_id")
    _EXTRA_ELEMENTS[eid] = element_dict


def get_extra_elements() -> dict[str, dict[str, Any]]:
    return dict(_EXTRA_ELEMENTS)


def known_rule_types() -> list[str]:
    builtins = ["numeric_range", "qualitative_enum", "qualitative_bool"]
    return sorted(set(builtins) | set(_RULE_TYPES))


def clear() -> None:
    _RULE_TYPES.clear()
    _EXTRA_ELEMENTS.clear()

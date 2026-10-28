"""Composite condition helpers (AllOf / AnyOf / Not / Leaf)."""

from __future__ import annotations

from typing import Any

from .types import CompositeCondition, Condition


def leaf(key: str, op: str, value: Any) -> CompositeCondition:
    return CompositeCondition(kind="leaf", leaf=Condition(key=key, op=op, value=value))


def all_of(*children: CompositeCondition) -> CompositeCondition:
    return CompositeCondition(kind="all_of", children=tuple(children))


def any_of(*children: CompositeCondition) -> CompositeCondition:
    return CompositeCondition(kind="any_of", children=tuple(children))


def not_(child: CompositeCondition) -> CompositeCondition:
    return CompositeCondition(kind="not", children=(child,))


def parse_composite(data: Any) -> CompositeCondition | None:
    """Parse a JSON dict into CompositeCondition.

    Formats:
      {"all_of": [{"key": ..., "op": ..., "value": ...}, ...]}
      {"any_of": [...]}
      {"not": {...}}
      {"key": ..., "op": ..., "value": ...}   # leaf
    """
    if data is None:
        return None
    if not isinstance(data, dict):
        raise ValueError(f"Composite condition must be dict, got {type(data)}")

    if "all_of" in data:
        children = tuple(parse_composite(c) for c in data["all_of"])
        children = tuple(c for c in children if c is not None)
        return CompositeCondition(kind="all_of", children=children)
    if "any_of" in data:
        children = tuple(parse_composite(c) for c in data["any_of"])
        children = tuple(c for c in children if c is not None)
        return CompositeCondition(kind="any_of", children=children)
    if "not" in data:
        inner = parse_composite(data["not"])
        return CompositeCondition(kind="not", children=(inner,) if inner else ())
    if "key" in data and "op" in data:
        return leaf(data["key"], data["op"], data.get("value"))
    raise ValueError(f"Unrecognized composite structure: {list(data.keys())}")

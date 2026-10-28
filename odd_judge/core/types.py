"""Core type definitions for ODD judgment."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal, Optional, Protocol


class ScenarioState(Enum):
    INSIDE = "INSIDE"
    BOUNDARY = "BOUNDARY"
    OUTSIDE = "OUTSIDE"
    UNDEFINED = "UNDEFINED"
    SKIPPED = "SKIPPED"


class Requirement(Enum):
    ALLOWED = "allowed"
    NOT_ALLOWED = "not_allowed"
    UNDEFINED = "undefined"


class DefinitionMode(Enum):
    PERMISSIVE = "permissive"
    RESTRICTIVE = "restrictive"
    DEFAULT = "default"


class Severity(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class ValueMissing:
    """Sentinel for missing observation values."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "ValueMissing"

    def __bool__(self) -> bool:
        return False


VALUE_MISSING = ValueMissing()


@dataclass(frozen=True)
class BoundaryMargin:
    mode: Literal["absolute", "relative"] = "absolute"
    value: float = 0.0

    def band_for_threshold(self, threshold: float) -> float:
        if self.mode == "absolute":
            return self.value
        return abs(threshold) * self.value

    def band_for_range(self, lo: float, hi: float) -> float:
        if self.mode == "absolute":
            return self.value
        return abs(hi - lo) * self.value


@dataclass(frozen=True)
class ThresholdSpec:
    op: str  # ">=", "<=", "between", "in", "not_in", "==", "!=", "bool"
    threshold: Optional[float] = None
    range: Optional[tuple[float, float]] = None
    values: Optional[tuple[Any, ...]] = None
    expect: Optional[bool] = None
    margin: Optional[BoundaryMargin] = None


@dataclass(frozen=True)
class Condition:
    key: str
    op: str
    value: Any


@dataclass(frozen=True)
class CompositeCondition:
    kind: Literal["all_of", "any_of", "not", "leaf"]
    children: tuple["CompositeCondition", ...] = ()
    leaf: Optional[Condition] = None


@dataclass(frozen=True)
class Rule:
    element_id: str
    path: tuple[str, ...]
    requirement: Requirement
    rule_type: str
    observation_key: str
    label_zh: str = ""
    qualifier: str = "include"
    spec: Optional[ThresholdSpec] = None
    allowed_values: Optional[tuple[Any, ...]] = None
    forbidden_values: Optional[tuple[Any, ...]] = None
    boundary_values: Optional[tuple[Any, ...]] = None
    expect: Optional[bool] = None
    applies_when: Optional[CompositeCondition] = None
    undefined_as: Optional[Requirement] = None
    warning_codes: dict[str, str] = field(default_factory=dict)
    tags: tuple[str, ...] = ()
    relation: Optional[dict[str, Any]] = None
    road_rule_match: Optional[str] = None
    confidence_threshold: Optional[float] = None
    confidence_band: Optional[float] = None


@dataclass(frozen=True)
class RuleResult:
    element_id: str
    state: ScenarioState
    observed: Any = None
    remaining_margin: Optional[float] = None
    margin_band: Optional[float] = None
    reason_code: Optional[str] = None
    confidence: float = 1.0
    message_zh: str = ""
    label_zh: str = ""
    requirement: str = ""
    rule_type: str = ""
    tags: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "element_id": self.element_id,
            "state": self.state.value,
            "observed": self.observed,
            "remaining_margin": self.remaining_margin,
            "margin_band": self.margin_band,
            "reason_code": self.reason_code,
            "confidence": self.confidence,
            "message_zh": self.message_zh,
            "label_zh": self.label_zh,
            "requirement": self.requirement,
            "rule_type": self.rule_type,
            "tags": list(self.tags),
        }
        return d


@dataclass
class Observation:
    fields: dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""
    source: str = ""

    def get(self, key: str) -> Any:
        """Resolve a dot-path key against nested fields."""
        parts = key.split(".")
        cur: Any = self.fields
        for p in parts:
            if isinstance(cur, dict) and p in cur:
                cur = cur[p]
            else:
                return VALUE_MISSING
        return cur

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Observation":
        if "fields" in data:
            return cls(
                fields=data.get("fields", {}),
                timestamp=data.get("timestamp", ""),
                source=data.get("source", ""),
            )
        return cls(fields=data, timestamp=data.get("timestamp", ""), source=data.get("source", ""))


@dataclass
class JudgmentReport:
    profile_id: str
    state: ScenarioState
    severity: Severity
    results: list[RuleResult] = field(default_factory=list)
    outside_reasons: list[str] = field(default_factory=list)
    boundary_reasons: list[str] = field(default_factory=list)
    undefined_handled: list[dict[str, Any]] = field(default_factory=list)
    skipped: list[dict[str, Any]] = field(default_factory=list)
    recommendation: str = "continue"
    active_switches: list[str] = field(default_factory=list)
    timestamp: str = ""
    definition_mode: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "state": self.state.value,
            "overall_severity": self.severity.value,
            "recommendation": self.recommendation,
            "definition_mode": self.definition_mode,
            "results": [r.to_dict() for r in self.results],
            "outside_reasons": self.outside_reasons,
            "boundary_reasons": self.boundary_reasons,
            "undefined_handled": self.undefined_handled,
            "skipped": self.skipped,
            "active_switches": self.active_switches,
            "timestamp": self.timestamp,
        }

    def to_json(self, indent: int = 2) -> str:
        import json

        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


@dataclass
class EvalContext:
    definition_mode: DefinitionMode = DefinitionMode.DEFAULT
    strict_undefined: bool = False


class RuleEvaluator(Protocol):
    def evaluate(self, rule: Rule, obs: Observation, ctx: EvalContext) -> RuleResult: ...

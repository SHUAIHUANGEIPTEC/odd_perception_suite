"""Core package exports."""

from .types import (
    BoundaryMargin,
    CompositeCondition,
    Condition,
    DefinitionMode,
    EvalContext,
    JudgmentReport,
    Observation,
    Requirement,
    Rule,
    RuleResult,
    ScenarioState,
    Severity,
    ThresholdSpec,
    VALUE_MISSING,
    ValueMissing,
)
from .evaluate import evaluate_rule, resolve_condition
from .aggregate import aggregate
from .composite import all_of, any_of, leaf, not_, parse_composite
from .margin import classify_numeric

__all__ = [
    "BoundaryMargin",
    "CompositeCondition",
    "Condition",
    "DefinitionMode",
    "EvalContext",
    "JudgmentReport",
    "Observation",
    "Requirement",
    "Rule",
    "RuleResult",
    "ScenarioState",
    "Severity",
    "ThresholdSpec",
    "VALUE_MISSING",
    "ValueMissing",
    "evaluate_rule",
    "resolve_condition",
    "aggregate",
    "all_of",
    "any_of",
    "leaf",
    "not_",
    "parse_composite",
    "classify_numeric",
]

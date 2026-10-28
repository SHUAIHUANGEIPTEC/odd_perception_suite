"""ODD Judgment Engine.

Evaluates current operational domain (COD) observations against an ODD profile
with quantitative and qualitative rules, producing INSIDE / BOUNDARY / OUTSIDE
judgments for boundary-proximity early warning.
"""

from .api import OddJudge, SwitchPolicy
from .core.types import (
    DefinitionMode,
    JudgmentReport,
    Observation,
    Requirement,
    Rule,
    RuleResult,
    ScenarioState,
    Severity,
)
from .config.loader import load_profile, load_profile_with_extends, load_switches
from .config.registry import register_element, register_rule_type

__version__ = "0.1.0"

__all__ = [
    "OddJudge",
    "SwitchPolicy",
    "DefinitionMode",
    "JudgmentReport",
    "Observation",
    "Requirement",
    "Rule",
    "RuleResult",
    "ScenarioState",
    "Severity",
    "load_profile",
    "load_profile_with_extends",
    "load_switches",
    "register_element",
    "register_rule_type",
    "__version__",
]

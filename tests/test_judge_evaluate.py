"""Tests for qualitative and quantitative rule evaluation."""

from odd_judge.core.evaluate import evaluate_rule
from odd_judge.core.types import (
    BoundaryMargin,
    DefinitionMode,
    EvalContext,
    Observation,
    Requirement,
    Rule,
    ScenarioState,
    ThresholdSpec,
)


def make_obs(**fields):
    return Observation(fields=fields)


def test_numeric_allowed_inside():
    rule = Rule(
        element_id="t.speed",
        path=("speed",),
        requirement=Requirement.ALLOWED,
        rule_type="numeric_range",
        observation_key="v.speed",
        spec=ThresholdSpec(op="between", range=(0, 80), margin=BoundaryMargin("absolute", 3)),
        warning_codes={"boundary": "B", "outside": "O"},
    )
    obs = make_obs(v={"speed": 50.0})
    r = evaluate_rule(rule, obs, EvalContext(DefinitionMode.RESTRICTIVE))
    assert r.state == ScenarioState.INSIDE


def test_numeric_allowed_boundary():
    rule = Rule(
        element_id="t.speed",
        path=("speed",),
        requirement=Requirement.ALLOWED,
        rule_type="numeric_range",
        observation_key="v.speed",
        spec=ThresholdSpec(op="between", range=(0, 80), margin=BoundaryMargin("absolute", 3)),
        warning_codes={"boundary": "B", "outside": "O"},
    )
    obs = make_obs(v={"speed": 78.5})
    r = evaluate_rule(rule, obs, EvalContext(DefinitionMode.RESTRICTIVE))
    assert r.state == ScenarioState.BOUNDARY
    assert r.reason_code == "B"


def test_numeric_allowed_outside():
    rule = Rule(
        element_id="t.speed",
        path=("speed",),
        requirement=Requirement.ALLOWED,
        rule_type="numeric_range",
        observation_key="v.speed",
        spec=ThresholdSpec(op="between", range=(0, 80), margin=BoundaryMargin("absolute", 3)),
        warning_codes={"boundary": "B", "outside": "O"},
    )
    obs = make_obs(v={"speed": 95.0})
    r = evaluate_rule(rule, obs, EvalContext(DefinitionMode.RESTRICTIVE))
    assert r.state == ScenarioState.OUTSIDE
    assert r.reason_code == "O"


def test_not_allowed_presence_is_outside():
    """requirement=not_allowed: forbidden condition PRESENT → OUTSIDE."""
    rule = Rule(
        element_id="t.intersection",
        path=("junction",),
        requirement=Requirement.NOT_ALLOWED,
        rule_type="qualitative_bool",
        observation_key="s.has_intersection",
        expect=True,
        warning_codes={"outside": "X"},
    )
    # forbidden condition present
    obs = make_obs(s={"has_intersection": True})
    r = evaluate_rule(rule, obs, EvalContext(DefinitionMode.RESTRICTIVE))
    assert r.state == ScenarioState.OUTSIDE


def test_not_allowed_absent_is_inside():
    rule = Rule(
        element_id="t.intersection",
        path=("junction",),
        requirement=Requirement.NOT_ALLOWED,
        rule_type="qualitative_bool",
        observation_key="s.has_intersection",
        expect=True,
        warning_codes={"outside": "X"},
    )
    obs = make_obs(s={"has_intersection": False})
    r = evaluate_rule(rule, obs, EvalContext(DefinitionMode.RESTRICTIVE))
    assert r.state == ScenarioState.INSIDE


def test_enum_allowed():
    rule = Rule(
        element_id="t.road",
        path=("road",),
        requirement=Requirement.ALLOWED,
        rule_type="qualitative_enum",
        observation_key="s.type",
        allowed_values=("expressway_main", "highway_main"),
        forbidden_values=("rural_road",),
        warning_codes={"outside": "R"},
    )
    ctx = EvalContext(DefinitionMode.RESTRICTIVE)
    assert evaluate_rule(rule, make_obs(s={"type": "expressway_main"}), ctx).state == ScenarioState.INSIDE
    assert evaluate_rule(rule, make_obs(s={"type": "rural_road"}), ctx).state == ScenarioState.OUTSIDE


def test_enum_boundary_values():
    rule = Rule(
        element_id="t.snow",
        path=("snow",),
        requirement=Requirement.ALLOWED,
        rule_type="qualitative_enum",
        observation_key="e.snow",
        allowed_values=("none", "light"),
        boundary_values=("moderate",),
        forbidden_values=("heavy",),
        warning_codes={"boundary": "SB", "outside": "SO"},
    )
    ctx = EvalContext(DefinitionMode.RESTRICTIVE)
    r = evaluate_rule(rule, make_obs(e={"snow": "moderate"}), ctx)
    assert r.state == ScenarioState.BOUNDARY
    assert r.reason_code == "SB"


def test_bool_allowed():
    rule = Rule(
        element_id="t.belt",
        path=("belt",),
        requirement=Requirement.ALLOWED,
        rule_type="qualitative_bool",
        observation_key="v.seatbelt",
        expect=True,
        warning_codes={"outside": "BE"},
    )
    ctx = EvalContext(DefinitionMode.RESTRICTIVE)
    assert evaluate_rule(rule, make_obs(v={"seatbelt": True}), ctx).state == ScenarioState.INSIDE
    assert evaluate_rule(rule, make_obs(v={"seatbelt": False}), ctx).state == ScenarioState.OUTSIDE


def test_undefined_requirement_skips():
    rule = Rule(
        element_id="t.foo",
        path=("foo",),
        requirement=Requirement.UNDEFINED,
        rule_type="qualitative_bool",
        observation_key="x.y",
        expect=True,
    )
    r = evaluate_rule(rule, make_obs(x={"y": True}), EvalContext(DefinitionMode.RESTRICTIVE))
    assert r.state == ScenarioState.UNDEFINED


def test_missing_obs_restrictive():
    rule = Rule(
        element_id="t.missing",
        path=("m",),
        requirement=Requirement.ALLOWED,
        rule_type="qualitative_bool",
        observation_key="x.nope",
        expect=True,
    )
    r = evaluate_rule(rule, make_obs(x={}), EvalContext(DefinitionMode.RESTRICTIVE))
    assert r.state == ScenarioState.OUTSIDE
    assert r.reason_code == "UNDEFINED-STRICT"


def test_missing_obs_permissive():
    rule = Rule(
        element_id="t.missing",
        path=("m",),
        requirement=Requirement.ALLOWED,
        rule_type="qualitative_bool",
        observation_key="x.nope",
        expect=True,
    )
    r = evaluate_rule(rule, make_obs(x={}), EvalContext(DefinitionMode.PERMISSIVE))
    assert r.state == ScenarioState.INSIDE


def test_missing_obs_default():
    rule = Rule(
        element_id="t.missing",
        path=("m",),
        requirement=Requirement.ALLOWED,
        rule_type="qualitative_bool",
        observation_key="x.nope",
        expect=True,
    )
    r = evaluate_rule(rule, make_obs(x={}), EvalContext(DefinitionMode.DEFAULT))
    assert r.state == ScenarioState.UNDEFINED

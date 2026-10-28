"""Tests for composite conditions and aggregation."""

from odd_judge.core.aggregate import aggregate
from odd_judge.core.composite import all_of, any_of, leaf, not_, parse_composite
from odd_judge.core.evaluate import resolve_condition
from odd_judge.core.types import Observation, RuleResult, ScenarioState, Severity


def test_all_of():
    obs = Observation(fields={"a": 1, "b": 2})
    cond = all_of(leaf("a", "==", 1), leaf("b", "==", 2))
    assert resolve_condition(cond, obs) is True
    cond2 = all_of(leaf("a", "==", 1), leaf("b", "==", 99))
    assert resolve_condition(cond2, obs) is False


def test_any_of():
    obs = Observation(fields={"a": 1})
    cond = any_of(leaf("a", "==", 99), leaf("a", "==", 1))
    assert resolve_condition(cond, obs) is True


def test_not():
    obs = Observation(fields={"a": 1})
    cond = not_(leaf("a", "==", 1))
    assert resolve_condition(cond, obs) is False
    cond2 = not_(leaf("a", "==", 2))
    assert resolve_condition(cond2, obs) is True


def test_parse_composite():
    data = {"all_of": [{"key": "x.y", "op": ">=", "value": 10}]}
    c = parse_composite(data)
    assert c is not None
    assert c.kind == "all_of"
    assert resolve_condition(c, Observation(fields={"x": {"y": 15}})) is True
    assert resolve_condition(c, Observation(fields={"x": {"y": 5}})) is False


def test_aggregate_worst_case():
    results = [
        RuleResult(element_id="a", state=ScenarioState.INSIDE),
        RuleResult(element_id="b", state=ScenarioState.BOUNDARY, message_zh="near", reason_code="B1"),
        RuleResult(element_id="c", state=ScenarioState.OUTSIDE, message_zh="out", reason_code="O1"),
    ]
    rep = aggregate("test", results)
    assert rep.state == ScenarioState.OUTSIDE
    assert rep.severity == Severity.CRITICAL
    assert rep.recommendation == "mrm_or_refuse"
    assert len(rep.outside_reasons) == 1


def test_aggregate_boundary_only():
    results = [
        RuleResult(element_id="a", state=ScenarioState.INSIDE),
        RuleResult(element_id="b", state=ScenarioState.BOUNDARY, message_zh="near", reason_code="B1"),
    ]
    rep = aggregate("test", results)
    assert rep.state == ScenarioState.BOUNDARY
    assert rep.severity == Severity.WARNING
    assert rep.recommendation == "warn_driver"
    assert len(rep.boundary_reasons) == 1


def test_aggregate_inside():
    results = [RuleResult(element_id="a", state=ScenarioState.INSIDE)]
    rep = aggregate("test", results)
    assert rep.state == ScenarioState.INSIDE
    assert rep.recommendation == "continue"

"""city_noa_cn override: intersections allowed, traffic light compliance judged."""

from pathlib import Path

from odd_judge.api import OddJudge
from odd_judge.config.loader import load_profile_with_extends
from odd_judge.core.types import Observation

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "profiles" / "byd_tjp.json"
OVERRIDE = ROOT / "profiles" / "overrides" / "city_noa_cn.json"

JUNCTION = "ODC.1.4.1.intersection"
LIGHT = "ODC.1.5.1.traffic_light_state"
ANNOTATION = "ODC.1.5.2.intersection_annotation"


def _judge(*, city: bool = True) -> OddJudge:
    pid, mode, rules = load_profile_with_extends(
        BASE, override_paths=[OVERRIDE] if city else None
    )
    return OddJudge(pid, mode, rules)


def _states(fields: dict, *, city: bool = True) -> dict[str, str]:
    report = _judge(city=city).evaluate(Observation(fields=fields))
    return {r.element_id: r.state.value for r in report.results}


def test_override_lifts_intersection_ban_and_adds_two_elements():
    rules = {r.element_id: r for r in _judge().rules}
    assert rules[JUNCTION].requirement.value == "undefined"
    assert LIGHT in rules
    assert ANNOTATION in rules


def test_tjp_keeps_intersection_forbidden():
    """Without the override the TJP semantics must not change."""
    assert _states({"scenery": {"has_intersection": True}}, city=False)[JUNCTION] == "OUTSIDE"


def test_light_states_when_applicable():
    fields = {"scenery": {"traffic_light_applicable": True}}
    for state, expected in (("green", "INSIDE"), ("yellow", "BOUNDARY"), ("red", "OUTSIDE")):
        obs = {"scenery": {**fields["scenery"], "traffic_light_state": state}}
        assert _states(obs)[LIGHT] == expected, state


def test_unknown_light_state_is_conservative():
    obs = {"scenery": {"traffic_light_applicable": True, "traffic_light_state": "unknown"}}
    assert _states(obs)[LIGHT] == "OUTSIDE"


def test_light_rule_skipped_when_not_applicable():
    """A red light for the crossing lane must not fail our ODD."""
    obs = {"scenery": {"traffic_light_applicable": False, "traffic_light_state": "red"}}
    assert _states(obs)[LIGHT] == "SKIPPED"
    assert _states({"scenery": {"traffic_light_state": "red"}})[LIGHT] == "SKIPPED"


def test_intersection_type_must_be_annotated():
    for kind in ("signalized", "unsignalized", "none"):
        assert _states({"scenery": {"intersection_type": kind}})[ANNOTATION] == "INSIDE", kind
    assert _states({})[ANNOTATION] == "OUTSIDE"

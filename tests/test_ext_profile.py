"""odc_ext_cn extension profile: the three added ODD dimensions, end to end."""

from pathlib import Path

from odd_judge.api import OddJudge
from odd_judge.config.loader import load_profile_with_extends
from odd_judge.core.types import Observation

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "profiles" / "byd_tjp.json"
EXT = ROOT / "profiles" / "odc_ext_cn.json"

FRICTION = "ODC.1.5.1.road_friction"
DENSITY = "ODC.4.2.1.traffic_density"
SPEED_LIMIT = "ODC.2.1.5.speed_limit"


def _judge() -> OddJudge:
    pid, mode, rules = load_profile_with_extends(BASE, extends_paths=[EXT])
    return OddJudge(pid, mode, rules)


def _states(fields: dict) -> dict[str, str]:
    report = _judge().evaluate(Observation(fields=fields))
    return {r.element_id: r.state.value for r in report.results}


def test_extension_adds_three_rules():
    judge = _judge()
    assert len(judge.rules) > 42
    for eid in (FRICTION, DENSITY, SPEED_LIMIT):
        assert eid in {r.element_id for r in judge.rules}


def test_friction_states_from_proxy():
    assert _states({"environment": {"friction_mu": 0.6}})[FRICTION] == "INSIDE"
    assert _states({"environment": {"friction_mu": 0.45}})[FRICTION] == "BOUNDARY"
    assert _states({"environment": {"friction_mu": 0.3}})[FRICTION] == "OUTSIDE"


def test_friction_missing_is_outside_under_restrictive():
    assert _states({})[FRICTION] == "OUTSIDE"


def test_density_states_from_vehicle_count():
    assert _states({"dynamic": {"surrounding_vehicles": 10}})[DENSITY] == "INSIDE"
    assert _states({"dynamic": {"surrounding_vehicles": 14}})[DENSITY] == "BOUNDARY"
    assert _states({"dynamic": {"surrounding_vehicles": 20}})[DENSITY] == "OUTSIDE"


def test_speed_limit_states_from_overspeed():
    assert _states({"vehicle": {"overspeed_kmh": 2.0}})[SPEED_LIMIT] == "INSIDE"
    assert _states({"vehicle": {"overspeed_kmh": 4.0}})[SPEED_LIMIT] == "BOUNDARY"
    assert _states({"vehicle": {"overspeed_kmh": 9.0}})[SPEED_LIMIT] == "OUTSIDE"


def test_speed_limit_missing_is_allowed_by_override():
    """No map sidecar -> no speed limit -> the element must not force OUTSIDE."""
    assert _states({})[SPEED_LIMIT] == "INSIDE"

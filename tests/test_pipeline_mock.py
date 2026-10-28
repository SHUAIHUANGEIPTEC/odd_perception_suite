"""Pipeline end-to-end with synthetic attr/detections + sidecar (no torch)."""

from tests.conftest import make_attr, make_det

from odd_suite.adapters.mock_adapter import build_demo_scenario
from odd_suite.pipeline.config import build_config
from odd_suite.pipeline.runner import Pipeline


def _pipeline(**kwargs) -> Pipeline:
    return Pipeline(build_config(**kwargs))


def test_demo_scenarios_reach_expected_states():
    pipeline = _pipeline()
    expected = {
        "highway_inside": "INSIDE",
        "curve_boundary": "BOUNDARY",
        "rain_outside": "OUTSIDE",
    }
    for name, state in expected.items():
        scenario = build_demo_scenario(name)
        result = pipeline.run(
            attr_override=scenario.attr,
            detections=scenario.detections,
            sidecar={"fields": scenario.sidecar_fields},
            timestamp=scenario.timestamp,
            source=f"test:{name}",
        )
        assert result.state == state, f"{name} -> {result.state}"


def test_mapped_env_fields_reach_observation():
    pipeline = _pipeline()
    scenario = build_demo_scenario("rain_outside")
    result = pipeline.run(
        attr_override=scenario.attr,
        detections=scenario.detections,
        sidecar={"fields": scenario.sidecar_fields},
    )
    assert result.observation.get("environment.visibility_m") == 40.0
    assert result.observation.get("environment.fog") is True
    assert result.observation.get("environment.illumination_lux") == 20.0
    assert result.observation.get("dynamic.obstacles_over_0_3m") == 4


def test_sidecar_wins_over_mapped_environment():
    pipeline = _pipeline()
    result = pipeline.run(
        attr_override=make_attr(weather=["Sunny"], visibility="GE500", lighting="Day"),
        sidecar={"fields": {"environment": {"visibility_m": 80.0}}},
    )
    assert result.observation.get("environment.visibility_m") == 80.0


def test_vehicle_traffic_does_not_trip_obstacle_rule():
    pipeline = _pipeline()
    scenario = build_demo_scenario("highway_inside")
    result = pipeline.run(
        attr_override=scenario.attr,
        detections=scenario.detections,
        sidecar={"fields": scenario.sidecar_fields},
    )
    assert result.observation.get("dynamic.obstacles_over_0_3m") == 0
    assert result.state == "INSIDE"


def test_pedestrian_flips_to_outside():
    pipeline = _pipeline()
    scenario = build_demo_scenario("highway_inside")
    result = pipeline.run(
        attr_override=scenario.attr,
        detections=[make_det("pedestrian")],
        sidecar={"fields": scenario.sidecar_fields},
    )
    assert result.observation.get("dynamic.pedestrians_present") is True
    assert result.state == "OUTSIDE"


def test_attr_disabled_leaves_environment_missing():
    from odd_judge.core.types import VALUE_MISSING

    pipeline = _pipeline(enable_attr=False)
    scenario = build_demo_scenario("highway_inside")
    result = pipeline.run(
        detections=[],
        sidecar={"fields": scenario.sidecar_fields},
    )
    assert result.observation.get("environment.visibility_m") is VALUE_MISSING

"""ObservationBuilder merge-precedence tests (no torch)."""

from odd_suite.mapping import ObservationBuilder
from odd_judge.core.types import VALUE_MISSING


def test_groups_land_in_fields():
    obs = (
        ObservationBuilder()
        .add("environment", {"visibility_m": 350.0})
        .add("dynamic", {"pedestrians_present": True})
        .build(timestamp="t0", source="test")
    )
    assert obs.get("environment.visibility_m") == 350.0
    assert obs.get("dynamic.pedestrians_present") is True
    assert obs.timestamp == "t0"
    assert obs.source == "test"


def test_sidecar_wins_over_mapping():
    obs = (
        ObservationBuilder()
        .add("environment", {"visibility_m": 2000.0, "fog": False})
        .add_sidecar({"fields": {"environment": {"visibility_m": 80.0}}})
        .build()
    )
    assert obs.get("environment.visibility_m") == 80.0  # instrument beats model
    assert obs.get("environment.fog") is False  # untouched keys survive


def test_missing_keys_resolve_to_value_missing():
    obs = ObservationBuilder().add("environment", {"fog": True}).build()
    assert obs.get("scenery.curve_radius_m") is VALUE_MISSING
    assert obs.get("environment.fog") is True


def test_later_add_overrides_same_group():
    obs = (
        ObservationBuilder()
        .add("sensors", {"confidence_camera": 0.5})
        .add("sensors", {"confidence_camera": 0.9})
        .build()
    )
    assert obs.get("sensors.confidence_camera") == 0.9


def test_overspeed_derived_from_speed_and_limit():
    obs = (
        ObservationBuilder()
        .add("vehicle", {"speed_kmh": 78.0})
        .add_sidecar({"vehicle": {"speed_limit_kmh": 60.0}})
        .build()
    )
    assert obs.get("vehicle.overspeed_kmh") == 18.0


def test_overspeed_absent_without_limit():
    obs = ObservationBuilder().add("vehicle", {"speed_kmh": 78.0}).build()
    assert obs.get("vehicle.overspeed_kmh") is VALUE_MISSING


def test_overspeed_derivation_does_not_mutate_group():
    builder = ObservationBuilder().add("vehicle", {"speed_kmh": 78.0})
    builder.add_sidecar({"vehicle": {"speed_limit_kmh": 60.0}}).build()
    assert "overspeed_kmh" not in builder._groups["vehicle"]

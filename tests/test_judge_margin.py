"""Tests for BoundaryMargin and classify_numeric."""

from odd_judge.core.margin import classify_numeric
from odd_judge.core.types import BoundaryMargin, ScenarioState, ThresholdSpec


def test_absolute_band():
    m = BoundaryMargin(mode="absolute", value=20.0)
    assert m.band_for_threshold(200.0) == 20.0
    assert m.band_for_range(100.0, 200.0) == 20.0


def test_relative_band():
    m = BoundaryMargin(mode="relative", value=0.1)
    assert m.band_for_threshold(200.0) == 20.0
    assert m.band_for_range(0.0, 100.0) == 10.0


def test_ge_inside():
    spec = ThresholdSpec(op=">=", threshold=200.0, margin=BoundaryMargin("absolute", 20.0))
    state, rem, band = classify_numeric(250.0, spec)
    assert state == ScenarioState.INSIDE
    assert rem == 50.0
    assert band == 20.0


def test_ge_boundary():
    spec = ThresholdSpec(op=">=", threshold=200.0, margin=BoundaryMargin("absolute", 20.0))
    state, rem, band = classify_numeric(212.0, spec)
    assert state == ScenarioState.BOUNDARY
    assert rem == 12.0


def test_ge_outside():
    spec = ThresholdSpec(op=">=", threshold=200.0, margin=BoundaryMargin("absolute", 20.0))
    state, rem, _ = classify_numeric(180.0, spec)
    assert state == ScenarioState.OUTSIDE
    assert rem == -20.0


def test_le_inside_boundary_outside():
    spec = ThresholdSpec(op="<=", threshold=9.0, margin=BoundaryMargin("absolute", 1.0))
    assert classify_numeric(5.0, spec)[0] == ScenarioState.INSIDE
    assert classify_numeric(8.5, spec)[0] == ScenarioState.BOUNDARY
    assert classify_numeric(10.0, spec)[0] == ScenarioState.OUTSIDE


def test_between_inside():
    spec = ThresholdSpec(op="between", range=(2.5, 5.5), margin=BoundaryMargin("absolute", 0.2))
    state, rem, _ = classify_numeric(3.5, spec)
    assert state == ScenarioState.INSIDE
    assert rem == 1.0  # min(3.5-2.5, 5.5-3.5) = 1.0


def test_between_boundary():
    spec = ThresholdSpec(op="between", range=(2.5, 5.5), margin=BoundaryMargin("absolute", 0.2))
    state, rem, _ = classify_numeric(2.65, spec)
    assert state == ScenarioState.BOUNDARY
    assert abs(rem - 0.15) < 1e-9


def test_between_outside():
    spec = ThresholdSpec(op="between", range=(2.5, 5.5), margin=BoundaryMargin("absolute", 0.2))
    state, rem, _ = classify_numeric(2.0, spec)
    assert state == ScenarioState.OUTSIDE
    assert rem == -0.5

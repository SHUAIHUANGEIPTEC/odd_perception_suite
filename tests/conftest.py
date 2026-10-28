"""Shared fixtures: frozen AttrResult / detections, mapping tables."""

from __future__ import annotations

import pytest

from odd_suite.mapping import DynamicRollup, EnvMapper, MappingTables, SensorDefaults
from odd_suite.schema import AttrResult, Detection3D


def make_attr(
    weather: list[str] | None = None,
    visibility: str = "GE500",
    lighting: str = "Day",
    road_condition: str = "Dry",
    road_material: str = "Asphalt",
) -> AttrResult:
    return AttrResult(
        image="test.jpg",
        prediction={
            "weather": weather or [],
            "visibility": visibility,
            "lighting": lighting,
            "road_condition": road_condition,
            "road_material": road_material,
        },
        probabilities={},
    )


def make_det(name: str, score: float = 0.9) -> Detection3D:
    return Detection3D(name=name, score=score)


@pytest.fixture(scope="session")
def tables() -> MappingTables:
    return MappingTables.load()


@pytest.fixture()
def env_mapper(tables) -> EnvMapper:
    return EnvMapper(tables)


@pytest.fixture()
def rollup(tables) -> DynamicRollup:
    return DynamicRollup(tables, score_threshold=0.3)


@pytest.fixture()
def sensors(tables) -> SensorDefaults:
    return SensorDefaults(tables)

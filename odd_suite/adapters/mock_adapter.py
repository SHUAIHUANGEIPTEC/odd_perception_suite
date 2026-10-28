"""Mock demo scenarios: synthetic model outputs exercising the full pipeline.

The base scenery/vehicle observations come from the complete sample observation
JSONs (``examples/observations/sample_*.json``); environment/dynamic/sensors are
produced by the mapping layer from a synthetic AttrResult + detection list, so
``odd-suite demo`` exercises attr mapping, detection rollup, observation
assembly and judgment without torch or GPUs.

(The ``odd_judge.perception.mock`` fixtures carry only a subset of scenery
fields; under ``definition_mode=restrictive`` the missing keys would force
UNDEFINED-STRICT -> OUTSIDE, hence the JSON samples are used instead.)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from odd_suite.paths import examples_dir
from odd_suite.schema import AttrResult, Detection3D

_DEMO_SCENARIO_FILES: dict[str, str] = {
    "highway_inside": "sample_highway_inside.json",
    "curve_boundary": "sample_curve_boundary.json",
    "rain_outside": "sample_rain_outside.json",
}

_DEMO_ATTRS: dict[str, dict[str, Any]] = {
    "highway_inside": {
        "weather": ["Sunny"],
        "visibility": "GE500",
        "lighting": "Day",
        "road_condition": "Dry",
        "road_material": "Asphalt",
    },
    "curve_boundary": {
        "weather": ["Cloudy"],
        "visibility": "200_500",
        "lighting": "Day",
        "road_condition": "Dry",
        "road_material": "Asphalt",
    },
    "rain_outside": {
        "weather": ["Rain", "Fog"],
        "visibility": "LT50",
        "lighting": "NightUnlit",
        "road_condition": "Wet",
        "road_material": "Asphalt",
    },
}

_DEMO_DETECTIONS: dict[str, list[tuple[str, float]]] = {
    "highway_inside": [("car", 0.92), ("car", 0.88), ("truck", 0.85)],
    "curve_boundary": [("car", 0.87), ("car", 0.81)],
    "rain_outside": [
        ("traffic_cone", 0.62),
        ("traffic_cone", 0.58),
        ("traffic_cone", 0.55),
        ("barrier", 0.60),
    ],
}


@dataclass(frozen=True)
class DemoScenario:
    name: str
    timestamp: str = ""
    sidecar_fields: dict[str, Any] = field(default_factory=dict)
    attr: AttrResult | None = None
    detections: list[Detection3D] = field(default_factory=list)


def _load_sample(name: str) -> dict[str, Any]:
    path = examples_dir() / "observations" / _DEMO_SCENARIO_FILES[name]
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def build_demo_scenario(name: str) -> DemoScenario:
    if name not in _DEMO_SCENARIO_FILES:
        raise ValueError(
            f"Unknown demo scenario: {name}. Available: {list_demo_scenarios()}"
        )
    sample = _load_sample(name)
    fields = sample.get("fields", {})
    # Only instrument-like groups come from the fixture; glare has no model head,
    # so the sidecar supplies it (mapping still drives visibility/rain/snow/lux).
    sidecar_fields: dict[str, Any] = {
        "scenery": fields.get("scenery", {}),
        "vehicle": fields.get("vehicle", {}),
        "environment": {"glare": fields.get("environment", {}).get("glare", False)},
    }
    attr_spec = _DEMO_ATTRS[name]
    attr = AttrResult(
        image=f"demo://{name}",
        prediction={
            "weather": attr_spec["weather"],
            "visibility": attr_spec["visibility"],
            "lighting": attr_spec["lighting"],
            "road_condition": attr_spec["road_condition"],
            "road_material": attr_spec["road_material"],
        },
        probabilities={},
    )
    detections = [
        Detection3D(name=cls, score=score) for cls, score in _DEMO_DETECTIONS[name]
    ]
    return DemoScenario(
        name=name,
        timestamp=str(sample.get("timestamp", "")),
        sidecar_fields=sidecar_fields,
        attr=attr,
        detections=detections,
    )


def list_demo_scenarios() -> list[str]:
    return list(_DEMO_SCENARIO_FILES)


def load_demo_observation(name: str) -> dict[str, Any]:
    """Complete sample observation JSON (all scenery/vehicle fields present)."""
    if name not in _DEMO_SCENARIO_FILES:
        raise ValueError(
            f"Unknown demo scenario: {name}. Available: {list_demo_scenarios()}"
        )
    return _load_sample(name)

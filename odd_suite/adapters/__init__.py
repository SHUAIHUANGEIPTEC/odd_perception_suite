"""Observation adapters and model facades."""

from odd_suite.adapters.attr_adapter import ImageAttrAdapter
from odd_suite.adapters.detect_adapter import (
    NuScenesDetAdapter,
    Sparse4DDetector,
    parse_nuscenes_predictions,
)
from odd_suite.adapters.mock_adapter import DemoScenario, build_demo_scenario, list_demo_scenarios
from odd_suite.adapters.sidecar_adapter import JsonSidecarAdapter

__all__ = [
    "DemoScenario",
    "ImageAttrAdapter",
    "JsonSidecarAdapter",
    "NuScenesDetAdapter",
    "Sparse4DDetector",
    "build_demo_scenario",
    "list_demo_scenarios",
    "parse_nuscenes_predictions",
]

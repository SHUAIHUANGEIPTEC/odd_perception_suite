"""Mock adapter with named scenario fixtures."""

from __future__ import annotations

from typing import Any

from ..core.types import Observation
from .base import ObservationAdapter

# Scenario fixtures: inside / boundary / outside
SCENARIOS: dict[str, dict[str, Any]] = {
    "highway_inside": {
        "timestamp": "2026-02-03T10:00:00Z",
        "source": "mock:highway_inside",
        "fields": {
            "scenery": {
                "road_type": "expressway_main",
                "curve_radius_m": 500.0,
                "lane_width_m": 3.5,
                "longitudinal_grade_pct": 3.0,
                "cross_slope_pct": 1.0,
                "marking_clear": True,
                "has_intersection": False,
                "construction_zone": False,
                "has_roundabout": False,
                "has_toll_station": False,
                "has_railway_crossing": False,
                "lane_count": 3,
            },
            "vehicle": {
                "speed_kmh": 72.0,
                "lat_accel_mps2": 0.5,
                "lon_accel_mps2": -0.5,
                "activation_speed_kmh": 72.0,
                "seatbelt_on": True,
                "driver_in_seat": True,
                "driver_eyes_closed_s": 0.0,
                "driver_attention_ok": True,
                "map_coverage": True,
                "hd_map_coverage": True,
                "ads_fault": False,
                "dssad_ok": True,
                "dms_ok": True,
            },
            "environment": {
                "visibility_m": 2000.0,
                "rain_mm_per_h": 0.0,
                "snow_level": "none",
                "illumination_lux": 5000,
                "glare": False,
                "fog": False,
            },
            "dynamic": {
                "vru_count_nearby": 0,
                "animals_over_0_3m": 0,
                "obstacles_over_0_3m": 0,
                "two_wheelers_present": False,
                "pedestrians_present": False,
                "construction_present": False,
            },
            "sensors": {
                "camera_degraded": False,
                "lidar_degraded": False,
                "confidence_camera": 0.95,
                "confidence_lidar": 0.92,
            },
        },
    },
    "curve_boundary": {
        "timestamp": "2026-02-03T10:05:00Z",
        "source": "mock:curve_boundary",
        "fields": {
            "scenery": {
                "road_type": "highway_main",
                "curve_radius_m": 212.0,
                "lane_width_m": 3.2,
                "longitudinal_grade_pct": 4.0,
                "cross_slope_pct": 1.5,
                "marking_clear": True,
                "has_intersection": False,
                "construction_zone": False,
                "has_roundabout": False,
                "has_toll_station": False,
                "has_railway_crossing": False,
                "lane_count": 2,
            },
            "vehicle": {
                "speed_kmh": 78.0,
                "lat_accel_mps2": 1.5,
                "lon_accel_mps2": -1.0,
                "activation_speed_kmh": 78.0,
                "seatbelt_on": True,
                "driver_in_seat": True,
                "driver_eyes_closed_s": 0.0,
                "driver_attention_ok": True,
                "map_coverage": True,
                "hd_map_coverage": True,
                "ads_fault": False,
                "dssad_ok": True,
                "dms_ok": True,
            },
            "environment": {
                "visibility_m": 250.0,
                "rain_mm_per_h": 0.0,
                "snow_level": "none",
                "illumination_lux": 1500,
                "glare": False,
                "fog": False,
            },
            "dynamic": {
                "vru_count_nearby": 0,
                "animals_over_0_3m": 0,
                "obstacles_over_0_3m": 0,
                "two_wheelers_present": False,
                "pedestrians_present": False,
                "construction_present": False,
            },
            "sensors": {
                "camera_degraded": False,
                "lidar_degraded": False,
                "confidence_camera": 0.88,
                "confidence_lidar": 0.85,
            },
        },
    },
    "rain_outside": {
        "timestamp": "2026-02-03T10:10:00Z",
        "source": "mock:rain_outside",
        "fields": {
            "scenery": {
                "road_type": "expressway_main",
                "curve_radius_m": 400.0,
                "lane_width_m": 3.5,
                "longitudinal_grade_pct": 2.0,
                "cross_slope_pct": 1.0,
                "marking_clear": False,
                "has_intersection": False,
                "construction_zone": False,
                "has_roundabout": False,
                "has_toll_station": False,
                "has_railway_crossing": False,
                "lane_count": 3,
            },
            "vehicle": {
                "speed_kmh": 60.0,
                "lat_accel_mps2": 0.5,
                "lon_accel_mps2": -2.0,
                "activation_speed_kmh": 60.0,
                "seatbelt_on": True,
                "driver_in_seat": True,
                "driver_eyes_closed_s": 0.0,
                "driver_attention_ok": True,
                "map_coverage": True,
                "hd_map_coverage": True,
                "ads_fault": False,
                "dssad_ok": True,
                "dms_ok": True,
            },
            "environment": {
                "visibility_m": 150.0,
                "rain_mm_per_h": 18.0,
                "snow_level": "none",
                "illumination_lux": 200,
                "glare": False,
                "fog": True,
            },
            "dynamic": {
                "vru_count_nearby": 0,
                "animals_over_0_3m": 0,
                "obstacles_over_0_3m": 0,
                "two_wheelers_present": False,
                "pedestrians_present": False,
                "construction_present": False,
            },
            "sensors": {
                "camera_degraded": True,
                "lidar_degraded": True,
                "confidence_camera": 0.55,
                "confidence_lidar": 0.50,
            },
        },
    },
}


class MockAdapter(ObservationAdapter):
    """Return canned scenario observations by name."""

    def __init__(self, scenario: str = "highway_inside"):
        if scenario not in SCENARIOS:
            raise ValueError(f"Unknown scenario: {scenario}. Available: {list(SCENARIOS)}")
        self._scenario = scenario

    def get_observation(self) -> Observation:
        return Observation.from_dict(SCENARIOS[self._scenario])

    @staticmethod
    def list_scenarios() -> list[str]:
        return list(SCENARIOS)

"""Observation ``sensors.*`` fields from detection scores or policy defaults."""

from __future__ import annotations

from typing import Any

from odd_suite.mapping.env_mapper import MappingTables
from odd_suite.schema import Detection3D


class SensorDefaults:
    def __init__(self, tables: MappingTables | None = None):
        self.tables = tables or MappingTables.load()
        self.policy = self.tables.sensor_policies

    def map(self, detections: list[Detection3D] | None) -> dict[str, Any]:
        policy = self.policy
        conf_cam = float(policy.get("default_confidence_camera", 0.95))
        conf_lidar = float(policy.get("default_confidence_lidar", 0.92))

        if (
            detections
            and policy.get("from_detections", True)
            and policy.get("mean_score_as_confidence", True)
            and len(detections) >= int(policy.get("min_detections_for_conf", 1))
        ):
            scores = [d.score for d in detections]
            conf_cam = max(0.0, min(1.0, sum(scores) / len(scores)))
            conf_lidar = conf_cam

        return {
            "camera_degraded": conf_cam < float(policy.get("camera_degraded_below", 0.3)),
            "lidar_degraded": conf_lidar < float(policy.get("lidar_degraded_below", 0.3)),
            "confidence_camera": conf_cam,
            "confidence_lidar": conf_lidar,
        }

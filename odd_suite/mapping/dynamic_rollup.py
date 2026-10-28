"""3D detection boxes -> Observation ``dynamic.*`` fields."""

from __future__ import annotations

from typing import Any

from odd_suite.mapping.env_mapper import MappingTables
from odd_suite.schema import Detection3D

_ROLE_FIELDS = {
    "vru": "pedestrians_present",
    "two_wheeler": "two_wheelers_present",
    "obstacle": "obstacles_over_0_3m",
    "animal": "animals_over_0_3m",
}


class DynamicRollup:
    """Aggregate detection boxes into ODD_Judge ``dynamic`` fields.

    Role assignment comes from the ``dynamic_roles`` table. The default table
    keeps ordinary vehicle traffic out of ``obstacles_over_0_3m`` (BYD TJP is a
    traffic-jam pilot: surrounding vehicles are expected, not obstacles).
    """

    def __init__(self, tables: MappingTables | None = None, score_threshold: float = 0.0):
        self.tables = tables or MappingTables.load()
        self.score_threshold = score_threshold

    def _role(self, name: str) -> str:
        roles = self.tables.dynamic_roles.get("roles", {})
        return roles.get(name, self.tables.dynamic_roles.get("default_role", "ignore"))

    def map(self, detections: list[Detection3D]) -> dict[str, Any]:
        counts = self.role_counts(detections)
        return {
            "pedestrians_present": counts.get("vru", 0) > 0,
            "two_wheelers_present": counts.get("two_wheeler", 0) > 0,
            "obstacles_over_0_3m": counts.get("obstacle", 0),
            "animals_over_0_3m": counts.get("animal", 0),
        }

    def role_counts(self, detections: list[Detection3D]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for det in detections:
            if det.score < self.score_threshold:
                continue
            role = self._role(det.name)
            if role == "ignore":
                continue
            counts[role] = counts.get(role, 0) + 1
        return counts

    def debug(self, detections: list[Detection3D]) -> dict[str, Any]:
        by_name: dict[str, int] = {}
        for det in detections:
            if det.score < self.score_threshold:
                continue
            by_name[det.name] = by_name.get(det.name, 0) + 1
        return {
            "per_class_counts": by_name,
            "role_counts": self.role_counts(detections),
            "score_threshold": self.score_threshold,
        }

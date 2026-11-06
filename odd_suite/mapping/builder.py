"""Merge multi-source field groups into one Observation.

Precedence: sidecar explicit values > explicit kwargs > mapping defaults > missing.
"""

from __future__ import annotations

from typing import Any

from odd_judge.core.types import Observation


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _with_derived(fields: dict[str, Any]) -> dict[str, Any]:
    """Add cross-signal fields the judge cannot compute from a single observation.

    ``overspeed_kmh = speed_kmh - speed_limit_kmh``: the judge compares one field
    against a threshold, so the speed-limit rule is expressed on the difference.
    Missing either input leaves the field absent, which the rule may declare as
    ``undefined_as``.
    """
    vehicle = fields.get("vehicle")
    if not isinstance(vehicle, dict):
        return fields
    speed = vehicle.get("speed_kmh")
    limit = vehicle.get("speed_limit_kmh")
    if speed is None or limit is None:
        return fields
    try:
        overspeed = float(speed) - float(limit)
    except (TypeError, ValueError):
        return fields
    derived = dict(fields)
    derived["vehicle"] = {**vehicle, "overspeed_kmh": overspeed}
    return derived


class ObservationBuilder:
    def __init__(self) -> None:
        self._groups: dict[str, dict[str, Any]] = {}
        self._sidecar: dict[str, Any] = {}

    def add(self, group: str, values: dict[str, Any] | None) -> "ObservationBuilder":
        if values:
            self._groups[group] = _deep_merge(self._groups.get(group, {}), values)
        return self

    def add_sidecar(self, sidecar: dict[str, Any] | None) -> "ObservationBuilder":
        if not sidecar:
            return self
        fields = sidecar.get("fields", sidecar)
        if isinstance(fields, dict):
            self._sidecar = _deep_merge(self._sidecar, fields)
        return self

    def build(self, *, timestamp: str = "", source: str = "") -> Observation:
        fields = dict(self._groups)
        fields = _deep_merge(fields, self._sidecar)
        return Observation(fields=_with_derived(fields), timestamp=timestamp, source=source)

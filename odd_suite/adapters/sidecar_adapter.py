"""Sidecar observation JSON -> ``scenery`` / ``vehicle`` (and overrides)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from odd_judge.core.types import Observation
from odd_judge.perception.base import ObservationAdapter


class JsonSidecarAdapter(ObservationAdapter):
    """Calibrated/external signals (map, CAN, DMS) not covered by the models."""

    def __init__(self, source: Path | dict[str, Any]):
        if isinstance(source, (str, Path)):
            with Path(source).open(encoding="utf-8") as f:
                self._data = json.load(f)
        else:
            self._data = source

    @property
    def fields(self) -> dict[str, Any]:
        return self._data.get("fields", self._data)

    def get_observation(self) -> Observation:
        return Observation(
            fields=self.fields,
            timestamp=str(self._data.get("timestamp", "")),
            source=str(self._data.get("source", "sidecar")),
        )

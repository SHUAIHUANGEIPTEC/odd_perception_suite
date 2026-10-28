"""JSON file / JSONL stream observation reader."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..core.errors import ObservationError
from ..core.types import Observation
from .base import ObservationAdapter


class JsonFileAdapter(ObservationAdapter):
    """Read a single JSON observation file, or JSONL (one object per line)."""

    def __init__(self, path: str | Path):
        self._path = Path(path)
        if not self._path.exists():
            raise ObservationError(f"Observation file not found: {self._path}")
        self._observations: list[Observation] = []
        self._load()

    def _load(self) -> None:
        text = self._path.read_text(encoding="utf-8").strip()
        if not text:
            raise ObservationError(f"Empty observation file: {self._path}")
        # Try whole-file JSON first
        try:
            data = json.loads(text)
            if isinstance(data, list):
                self._observations = [Observation.from_dict(d) for d in data]
            else:
                self._observations = [Observation.from_dict(data)]
            return
        except json.JSONDecodeError:
            pass
        # JSONL fallback
        for i, line in enumerate(text.splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                self._observations.append(Observation.from_dict(json.loads(line)))
            except json.JSONDecodeError as e:
                raise ObservationError(f"Invalid JSONL at line {i + 1}: {e}") from e

    def get_observation(self) -> Observation:
        if not self._observations:
            raise ObservationError("No observations loaded")
        return self._observations[0]

    def get_stream(self) -> list[Observation]:
        return list(self._observations)


def load_observation_file(path: str | Path) -> Observation:
    return JsonFileAdapter(path).get_observation()


def load_observation_stream(path: str | Path) -> list[Observation]:
    return JsonFileAdapter(path).get_stream()

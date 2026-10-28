"""Perception adapter base class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..core.types import Observation


class ObservationAdapter(ABC):
    """Abstract adapter that produces Observation objects from a perception source."""

    @abstractmethod
    def get_observation(self) -> Observation:
        """Return a single observation snapshot."""

    def get_stream(self) -> list[Observation]:
        """Return a batch of observations (default: single)."""
        return [self.get_observation()]


class DictAdapter(ObservationAdapter):
    """Adapter wrapping a plain dict."""

    def __init__(self, data: dict[str, Any]):
        self._data = data

    def get_observation(self) -> Observation:
        return Observation.from_dict(self._data)

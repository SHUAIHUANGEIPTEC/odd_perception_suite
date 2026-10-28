"""Adapter base types: re-exported ObservationAdapter + model protocols."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from odd_judge.perception.base import DictAdapter, ObservationAdapter
from odd_suite.schema import AttrResult, Detection3D

__all__ = ["AttrModel", "DetectionParser", "DictAdapter", "Detector3D", "ObservationAdapter"]


@runtime_checkable
class AttrModel(Protocol):
    def predict(self, image: Path) -> AttrResult: ...


@runtime_checkable
class Detector3D(Protocol):
    def detect(self, *, sample_token: str | None = None) -> list[Detection3D]: ...


@runtime_checkable
class DetectionParser(Protocol):
    def parse(self, path: Path) -> dict[str, list[Detection3D]]: ...

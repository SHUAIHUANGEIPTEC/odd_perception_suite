"""Typed result objects shared across mapping, adapters and pipeline.

Pure dataclasses — no third-party imports (the judge stack must stay zero-dep).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from odd_judge.core.types import JudgmentReport, Observation


@dataclass(frozen=True)
class AttrResult:
    """Five-task ODD attribute classification output for one image."""

    image: str
    prediction: dict[str, Any]  # weather: list[str]; other tasks: str
    probabilities: dict[str, dict[str, float]]  # task -> class -> p
    checkpoint_epoch: int | None = None

    @property
    def weather(self) -> list[str]:
        value = self.prediction.get("weather", [])
        return list(value) if isinstance(value, list) else [str(value)]

    @property
    def visibility(self) -> str:
        return str(self.prediction.get("visibility", ""))

    @property
    def lighting(self) -> str:
        return str(self.prediction.get("lighting", ""))

    @property
    def road_condition(self) -> str:
        return str(self.prediction.get("road_condition", ""))

    @property
    def road_material(self) -> str:
        return str(self.prediction.get("road_material", ""))


@dataclass(frozen=True)
class Detection3D:
    """One 3D detection box in nuScenes submission layout."""

    name: str
    score: float
    sample_token: str = ""
    translation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    size: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    velocity: tuple[float, float] = (0.0, 0.0)
    attribute_name: str = ""

    @classmethod
    def from_nuscenes_record(cls, record: dict[str, Any]) -> "Detection3D":
        return cls(
            name=str(record.get("detection_name", "")),
            score=float(record.get("detection_score", 0.0)),
            sample_token=str(record.get("sample_token", "")),
            translation=tuple(record.get("translation", (0.0, 0.0, 0.0))),
            size=tuple(record.get("size", (0.0, 0.0, 0.0))),
            rotation=tuple(record.get("rotation", (1.0, 0.0, 0.0, 0.0))),
            velocity=tuple(record.get("velocity", (0.0, 0.0))),
            attribute_name=str(record.get("attribute_name", "")),
        )


@dataclass(frozen=True)
class PipelineConfig:
    profile: Path
    switches: Path | None = None
    extends: tuple[Path, ...] = ()
    overrides: tuple[Path, ...] = ()
    attr_checkpoint: Path | None = None
    detect_ckpt: Path | None = None
    device: str = "cpu"
    weather_threshold: float = 0.5
    score_threshold: float = 0.0
    mapping_dir: Path | None = None
    enable_attr: bool = True
    enable_detect: bool = False
    sidecar: Path | None = None


@dataclass(frozen=True)
class PipelineResult:
    observation: Observation
    judgment: JudgmentReport
    attr: AttrResult | None = None
    detections: list[Detection3D] | None = None
    debug: dict[str, Any] = field(default_factory=dict)

    @property
    def state(self) -> str:
        return self.judgment.state.value

    @property
    def recommendation(self) -> str:
        return self.judgment.recommendation

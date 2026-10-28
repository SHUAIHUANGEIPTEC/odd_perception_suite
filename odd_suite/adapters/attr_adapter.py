"""Image -> ODD attribute classification -> ``environment`` Observation group."""

from __future__ import annotations

from pathlib import Path

from odd_judge.core.types import Observation
from odd_judge.perception.base import ObservationAdapter
from odd_suite.mapping.env_mapper import EnvMapper, MappingTables
from odd_suite.schema import AttrResult


class ImageAttrAdapter(ObservationAdapter):
    """Wraps the vendored ODD attribute classifier (torch imported lazily)."""

    def __init__(
        self,
        checkpoint: Path,
        *,
        device: str = "cpu",
        weather_threshold: float = 0.5,
        tables: MappingTables | None = None,
        image: Path | None = None,
    ):
        self._checkpoint = Path(checkpoint)
        self._device = device
        self._weather_threshold = weather_threshold
        self._mapper = EnvMapper(tables)
        self._image = Path(image) if image else None
        self._classifier = None
        self.last_result: AttrResult | None = None

    def _load(self):
        if self._classifier is None:
            from odd_suite.vendor.attr.checkpoint import AttrClassifier

            self._classifier = AttrClassifier(
                self._checkpoint, self._device, self._weather_threshold
            )
        return self._classifier

    def predict(self, image: Path) -> AttrResult:
        self._image = Path(image)
        self.last_result = self._load().predict(self._image)
        return self.last_result

    def get_observation(self) -> Observation:
        if self.last_result is None:
            if self._image is None:
                raise ValueError("ImageAttrAdapter: call predict(image) first")
            self.predict(self._image)
        env = self._mapper.map(self.last_result)
        return Observation(fields={"environment": env}, source="attr_classifier")

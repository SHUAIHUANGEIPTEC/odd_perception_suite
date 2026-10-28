"""Pipeline orchestration: attrs + detections + sidecar -> Observation -> judgment."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from odd_judge.api import OddJudge
from odd_suite.adapters.attr_adapter import ImageAttrAdapter
from odd_suite.adapters.detect_adapter import NuScenesDetAdapter
from odd_suite.adapters.sidecar_adapter import JsonSidecarAdapter
from odd_suite.errors import ConfigError
from odd_suite.mapping.builder import ObservationBuilder
from odd_suite.mapping.env_mapper import EnvMapper
from odd_suite.mapping.registry import MapperRegistry
from odd_suite.schema import AttrResult, Detection3D, PipelineConfig, PipelineResult


class Pipeline:
    def __init__(self, config: PipelineConfig, judge: OddJudge | None = None):
        self.config = config
        self._judge = judge
        self.tables = MapperRegistry.load_tables(config.mapping_dir)
        self.env_mapper = EnvMapper(self.tables)

    def judge(self) -> OddJudge:
        if self._judge is None:
            self._judge = OddJudge.from_files(
                self.config.profile,
                self.config.switches,
                extends_paths=list(self.config.extends) or None,
                override_paths=list(self.config.overrides) or None,
            )
        return self._judge

    def run(
        self,
        *,
        images: list[Path] | None = None,
        detections: list[Detection3D] | None = None,
        sidecar: dict[str, Any] | Path | None = None,
        attr_override: AttrResult | None = None,
        timestamp: str = "",
        source: str = "",
    ) -> PipelineResult:
        builder = ObservationBuilder()
        debug: dict[str, Any] = {}
        attr_result: AttrResult | None = attr_override

        if attr_result is None and images and self.config.enable_attr:
            if self.config.attr_checkpoint is None:
                raise ConfigError(
                    "No attr checkpoint available; pass --checkpoint or disable attr"
                )
            adapter = ImageAttrAdapter(
                self.config.attr_checkpoint,
                device=self.config.device,
                weather_threshold=self.config.weather_threshold,
                tables=self.tables,
                image=images[0],
            )
            attr_result = adapter.predict(images[0])
            if len(images) > 1:
                debug["additional_images"] = [str(p) for p in images[1:]]
            debug["attr_mode"] = "model"
        elif attr_result is not None:
            debug["attr_mode"] = "override"

        if attr_result is not None:
            env = self.env_mapper.map(attr_result)
            env = MapperRegistry.apply_env_rules(attr_result, env)
            builder.add("environment", env)
            debug["environment_map"] = env
            debug["road"] = self.env_mapper.debug(attr_result)

        det_adapter: NuScenesDetAdapter | None = None
        if detections is not None:
            det_adapter = NuScenesDetAdapter(
                detections,
                tables=self.tables,
                score_threshold=self.config.score_threshold,
            )
            obs_part = det_adapter.get_observation()
            builder.add("dynamic", obs_part.fields.get("dynamic"))
            builder.add("sensors", obs_part.fields.get("sensors"))
            debug["detections"] = det_adapter.debug()

        if sidecar is not None:
            side_fields = (
                JsonSidecarAdapter(sidecar).fields
                if isinstance(sidecar, (str, Path))
                else sidecar.get("fields", sidecar)
            )
            builder.add_sidecar(side_fields)
            debug["sidecar_groups"] = (
                sorted(side_fields.keys()) if isinstance(side_fields, dict) else []
            )

        observation = builder.build(timestamp=timestamp, source=source)
        judgment = self.judge().evaluate(observation)
        return PipelineResult(
            observation=observation,
            judgment=judgment,
            attr=attr_result,
            detections=detections,
            debug=debug,
        )

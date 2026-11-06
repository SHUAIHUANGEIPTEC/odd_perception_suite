"""ODD attribute classification output -> Observation ``environment.*`` fields."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from odd_suite.paths import mapping_tables_dir
from odd_suite.schema import AttrResult

_TABLE_FILES = {
    "weather": "weather_env.json",
    "visibility": "visibility_env.json",
    "lighting": "lighting_env.json",
    "friction": "friction_env.json",
}


@dataclass(frozen=True)
class MappingTables:
    weather: dict[str, Any]
    visibility: dict[str, Any]
    lighting: dict[str, Any]
    dynamic_roles: dict[str, Any]
    sensor_policies: dict[str, Any]
    friction: dict[str, Any]

    @classmethod
    def load(cls, directory: Path | None = None) -> "MappingTables":
        base = Path(directory) if directory else mapping_tables_dir()

        def read(name: str) -> dict[str, Any]:
            path = base / name
            with path.open(encoding="utf-8") as f:
                return json.load(f)

        return cls(
            weather=read(_TABLE_FILES["weather"]),
            visibility=read(_TABLE_FILES["visibility"]),
            lighting=read(_TABLE_FILES["lighting"]),
            dynamic_roles=read("dynamic_roles.json"),
            sensor_policies=read("sensor_policies.json"),
            friction=read(_TABLE_FILES["friction"]),
        )


class EnvMapper:
    """Map AttrResult onto ODD_Judge ``environment`` fields.

    Multi-label weather items trigger independently; clamps (fog / haze / dust)
    are applied after the visibility band so contradictory readings resolve
    toward the degraded value.
    """

    def __init__(self, tables: MappingTables | None = None):
        self.tables = tables or MappingTables.load()

    def map(self, attr: AttrResult) -> dict[str, Any]:
        weather = self.tables.weather
        env: dict[str, Any] = dict(weather.get("baseline", {}))

        for label in attr.weather:
            rule = weather.get("rules", {}).get(label)
            if not rule:
                continue
            env.update(rule.get("set", {}))

        visibility = self.tables.visibility.get(attr.visibility)
        if visibility is not None:
            env["visibility_m"] = float(visibility)

        if "visibility_m" in env:
            if env.get("fog"):
                env["visibility_m"] = min(
                    env["visibility_m"], float(weather.get("fog_clamp_visibility_m", 350))
                )
            for label in attr.weather:
                clamp = weather.get("rules", {}).get(label, {}).get("clamp_visibility_m")
                if clamp is not None:
                    env["visibility_m"] = min(env["visibility_m"], float(clamp))

        illumination = self.tables.lighting.get(attr.lighting)
        if illumination is not None:
            env["illumination_lux"] = float(illumination)

        friction = self._friction_mu(attr)
        if friction is not None:
            env["friction_mu"] = friction

        return env

    def _friction_mu(self, attr: AttrResult) -> float | None:
        """Adhesion proxy from the road_condition / road_material heads.

        The classifier only names a surface class, so this is a table lookup, not
        a measurement. Unknown classes stay missing on purpose: the judge then
        reports the element as undefined instead of inventing a friction value.
        """
        tables = self.tables.friction
        mu = tables.get("condition_mu", {}).get(attr.road_condition)
        if mu is None:
            return None
        cap = tables.get("material_cap", {}).get(attr.road_material)
        return min(float(mu), float(cap)) if cap is not None else float(mu)

    def debug(self, attr: AttrResult) -> dict[str, Any]:
        """Non-judged extras (road_condition / road_material) for PipelineResult.debug."""
        return {
            "road_condition": attr.road_condition,
            "road_material": attr.road_material,
        }

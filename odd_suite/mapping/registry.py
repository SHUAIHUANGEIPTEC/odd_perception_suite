"""Extension points for custom mapping rules and detection class roles."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from odd_suite.mapping.env_mapper import MappingTables
from odd_suite.schema import AttrResult, Detection3D

EnvRuleFn = Callable[[AttrResult, dict[str, Any]], dict[str, Any]]


class MapperRegistry:
    """Registry aligned with ``odd_judge.config.registry`` plugin style."""

    _env_rules: dict[str, EnvRuleFn] = {}
    _role_overrides: dict[str, str] = {}

    @classmethod
    def register_env_rule(cls, name: str, fn: EnvRuleFn) -> None:
        cls._env_rules[name] = fn

    @classmethod
    def register_role(cls, nusc_class: str, role: str) -> None:
        cls._role_overrides[nusc_class] = role

    @classmethod
    def apply_env_rules(cls, attr: AttrResult, env: dict[str, Any]) -> dict[str, Any]:
        merged = dict(env)
        for fn in cls._env_rules.values():
            merged.update(fn(attr, merged))
        return merged

    @classmethod
    def apply_role_overrides(cls, tables: MappingTables) -> MappingTables:
        if not cls._role_overrides:
            return tables
        roles = dict(tables.dynamic_roles.get("roles", {}))
        roles.update(cls._role_overrides)
        dynamic = dict(tables.dynamic_roles)
        dynamic["roles"] = roles
        return MappingTables(
            weather=tables.weather,
            visibility=tables.visibility,
            lighting=tables.lighting,
            dynamic_roles=dynamic,
            sensor_policies=tables.sensor_policies,
            friction=tables.friction,
        )

    @classmethod
    def load_tables(cls, directory: Path | None = None) -> MappingTables:
        return cls.apply_role_overrides(MappingTables.load(directory))

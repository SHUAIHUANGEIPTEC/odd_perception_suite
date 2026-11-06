"""Pipeline configuration: defaults from configs/pipeline_default.json + CLI overrides."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from odd_suite.errors import ConfigError
from odd_suite.paths import configs_dir, profiles_dir, switches_dir, weights_dir
from odd_suite.schema import PipelineConfig


def default_config_path() -> Path:
    return configs_dir() / "pipeline_default.json"


def load_defaults(path: Path | None = None) -> dict[str, Any]:
    config_path = Path(path) if path else default_config_path()
    if not config_path.exists():
        return {}
    with config_path.open(encoding="utf-8") as f:
        return json.load(f)


def resolve_profile(value: str | Path | None) -> Path:
    """Accept a profile name (``byd_tjp``) or a path."""
    if value is None:
        value = "byd_tjp"
    candidate = Path(value)
    if candidate.exists():
        return candidate
    named = profiles_dir() / f"{value}.json"
    if named.exists():
        return named
    raise ConfigError(f"Profile not found: {value}")


def resolve_override(value: str | Path) -> Path:
    """Accept an override path or a name under ``profiles/overrides/``."""
    candidate = Path(value)
    if candidate.exists():
        return candidate
    named = profiles_dir() / "overrides" / f"{candidate.stem}.json"
    if named.exists():
        return named
    raise ConfigError(f"Override profile not found: {value}")


def resolve_switches(value: str | Path | None) -> Path | None:
    if value is None:
        default = switches_dir() / "default_switches.json"
        return default if default.exists() else None
    candidate = Path(value)
    if candidate.exists():
        return candidate
    named = switches_dir() / f"{value}.json"
    if named.exists():
        return named
    raise ConfigError(f"Switches file not found: {value}")


def build_config(
    *,
    profile: str | Path | None = None,
    switches: str | Path | None = None,
    attr_checkpoint: str | Path | None = None,
    detect_ckpt: str | Path | None = None,
    device: str | None = None,
    weather_threshold: float | None = None,
    score_threshold: float | None = None,
    sidecar: str | Path | None = None,
    mapping_dir: str | Path | None = None,
    extends: list[str | Path] | None = None,
    overrides: list[str | Path] | None = None,
    enable_attr: bool | None = None,
    enable_detect: bool | None = None,
    defaults_path: Path | None = None,
) -> PipelineConfig:
    defaults = load_defaults(defaults_path)

    def pick(explicit, key, fallback=None):
        return explicit if explicit is not None else defaults.get(key, fallback)

    attr_ckpt = pick(attr_checkpoint, "attr_checkpoint")
    if attr_ckpt is None:
        default_weight = weights_dir() / "attr_best.pth"
        attr_ckpt = default_weight if default_weight.exists() else None
    detect_ckpt = pick(detect_ckpt, "detect_ckpt")
    if detect_ckpt is None:
        default_weight = weights_dir() / "sparse4d_last.ckpt"
        detect_ckpt = default_weight if default_weight.exists() else None
    mapping = pick(mapping_dir, "mapping_dir")
    sidecar_value = pick(sidecar, "sidecar")

    return PipelineConfig(
        profile=resolve_profile(pick(profile, "profile")),
        switches=resolve_switches(pick(switches, "switches")),
        attr_checkpoint=Path(attr_ckpt) if attr_ckpt else None,
        detect_ckpt=Path(detect_ckpt) if detect_ckpt else None,
        device=str(pick(device, "device", "cpu")),
        weather_threshold=float(pick(weather_threshold, "weather_threshold", 0.5)),
        score_threshold=float(pick(score_threshold, "score_threshold", 0.0)),
        mapping_dir=Path(mapping) if mapping else None,
        extends=tuple(resolve_profile(p) for p in (extends or [])),
        overrides=tuple(resolve_override(p) for p in (overrides or [])),
        enable_attr=bool(pick(enable_attr, "enable_attr", True)),
        enable_detect=bool(pick(enable_detect, "enable_detect", False)),
        sidecar=Path(sidecar_value) if sidecar_value else None,
    )

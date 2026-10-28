"""Tests for config loading and switches."""

import json
from pathlib import Path

import pytest

from odd_judge.config.loader import load_profile, load_profile_with_extends, load_switches, parse_profile
from odd_judge.config.schema import validate_profile_dict
from odd_judge.core.errors import ConfigError

ROOT = Path(__file__).resolve().parent.parent
PROFILES = ROOT / "profiles"


def test_load_byd_profile():
    pid, mode, rules = load_profile(PROFILES / "byd_tjp.json")
    assert pid == "byd_tjp"
    assert mode.value == "restrictive"
    assert len(rules) > 30


def test_load_skeleton():
    pid, mode, rules = load_profile(PROFILES / "odc_skeleton.json")
    assert pid == "odc_skeleton"
    assert mode.value == "default"


def test_profile_validation_errors():
    bad = {"profile_id": "x", "elements": [{"element_id": "a", "requirement": "bad_req"}]}
    errors = validate_profile_dict(bad)
    assert any("invalid requirement" in e for e in errors)


def test_profile_validation_ok():
    with open(PROFILES / "byd_tjp.json", encoding="utf-8") as f:
        data = json.load(f)
    assert validate_profile_dict(data) == []


def test_override_merge():
    pid, mode, rules = load_profile_with_extends(
        PROFILES / "byd_tjp.json",
        override_paths=[PROFILES / "overrides" / "example_enterprise.json"],
    )
    ids = {r.element_id for r in rules}
    assert "ODC.ENT.1.road_temperature" in ids
    assert "ODC.1.2.1.curve_radius" in ids


def test_load_switches():
    sw = load_switches(ROOT / "switches" / "default_switches.json")
    assert sw["global_enabled"] is True
    assert sw["boundary_warnings_enabled"] is True


def test_parse_profile_from_dict():
    data = {
        "profile_id": "mini",
        "definition_mode": "permissive",
        "elements": [
            {
                "element_id": "x1",
                "requirement": "allowed",
                "rule_type": "numeric_range",
                "observation_key": "a.b",
                "op": ">=",
                "threshold": 1.0,
            }
        ],
    }
    pid, mode, rules = parse_profile(data)
    assert pid == "mini"
    assert len(rules) == 1

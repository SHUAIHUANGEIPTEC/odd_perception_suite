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


def _mini_profile(warning_codes_by_id: dict) -> dict:
    """Build a minimal valid profile with one element per given warning_codes."""
    return {
        "profile_id": "mini",
        "elements": [
            {
                "element_id": eid,
                "requirement": "allowed",
                "rule_type": "qualitative_bool",
                "observation_key": "a.b",
                "expect": True,
                "warning_codes": wc,
            }
            for eid, wc in warning_codes_by_id.items()
        ],
    }


def test_warning_codes_invalid_key():
    data = _mini_profile({"x1": {"warn": "X-001"}})
    errors = validate_profile_dict(data)
    assert any("invalid warning_codes key" in e for e in errors)


def test_warning_codes_invalid_format():
    data = _mini_profile({"x1": {"outside": "lower case code"}})
    errors = validate_profile_dict(data)
    assert any("invalid warning code format" in e for e in errors)


def test_warning_codes_duplicate_across_elements():
    data = _mini_profile(
        {
            "x1": {"outside": "X-001", "boundary": "X-NEAR"},
            "x2": {"boundary": "X-NEAR"},
        }
    )
    errors = validate_profile_dict(data)
    assert any("duplicate warning code" in e and "X-NEAR" in e for e in errors)


def test_warning_codes_valid():
    data = _mini_profile(
        {
            "x1": {"outside": "X-001", "boundary": "X-NEAR"},
            "x2": {"outside": "X-002"},
        }
    )
    assert validate_profile_dict(data) == []


def test_warning_codes_duplicate_via_registry_element():
    """Plugin-registered elements must not collide with profile codes."""
    from odd_judge.config import registry

    data = _mini_profile({"x1": {"outside": "X-001"}})
    registry.register_element(
        {
            "element_id": "plug.1",
            "requirement": "allowed",
            "rule_type": "qualitative_bool",
            "observation_key": "a.b",
            "expect": True,
            "warning_codes": {"boundary": "X-001"},
        }
    )
    try:
        with pytest.raises(ConfigError, match="duplicate warning code"):
            parse_profile(data)
    finally:
        registry.clear()


def test_warning_codes_doc_up_to_date():
    """docs/warning_codes.md code tables must match profiles/ (no drift)."""
    import subprocess
    import sys

    script = ROOT / "scripts" / "gen_warning_codes_doc.py"
    res = subprocess.run(
        [sys.executable, str(script), "--check"],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, res.stdout + res.stderr

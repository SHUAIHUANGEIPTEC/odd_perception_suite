"""Tests for perception adapters and CLI end-to-end."""

import json
import subprocess
import sys
from pathlib import Path

from odd_judge.api import OddJudge
from odd_judge.perception.json_stream import load_observation_file, load_observation_stream
from odd_judge.perception.mock import MockAdapter, SCENARIOS

ROOT = Path(__file__).resolve().parent.parent


def test_mock_adapter_scenarios():
    assert set(MockAdapter.list_scenarios()) == set(SCENARIOS)
    for name in SCENARIOS:
        obs = MockAdapter(name).get_observation()
        assert obs.fields


def test_json_file_adapter():
    obs = load_observation_file(ROOT / "examples" / "observations" / "sample_highway_inside.json")
    assert obs.get("scenery.road_type") == "expressway_main"


def test_judge_inside():
    judge = OddJudge.from_files(
        ROOT / "profiles" / "byd_tjp.json",
        switches_path=ROOT / "switches" / "default_switches.json",
    )
    report = judge.evaluate_path(ROOT / "examples" / "observations" / "sample_highway_inside.json")
    assert report.state.value == "INSIDE"
    assert report.recommendation == "continue"


def test_judge_boundary():
    judge = OddJudge.from_files(
        ROOT / "profiles" / "byd_tjp.json",
        switches_path=ROOT / "switches" / "default_switches.json",
    )
    report = judge.evaluate_path(ROOT / "examples" / "observations" / "sample_curve_boundary.json")
    assert report.state.value == "BOUNDARY"
    assert report.recommendation == "warn_driver"
    assert len(report.boundary_reasons) > 0


def test_judge_outside():
    judge = OddJudge.from_files(
        ROOT / "profiles" / "byd_tjp.json",
        switches_path=ROOT / "switches" / "default_switches.json",
    )
    report = judge.evaluate_path(ROOT / "examples" / "observations" / "sample_rain_outside.json")
    assert report.state.value == "OUTSIDE"
    assert report.recommendation == "mrm_or_refuse"
    assert len(report.outside_reasons) > 0


def test_switch_skips_rule():
    switches = {
        "global_enabled": True,
        "by_element": {"ODC.3.1.2.rain_intensity": False},
        "by_rule_type": {},
        "by_tag": {},
        "boundary_warnings_enabled": True,
    }
    judge = OddJudge.from_files(
        ROOT / "profiles" / "byd_tjp.json",
        switches_path=None,
    )
    judge.switch_policy.by_element["ODC.3.1.2.rain_intensity"] = False
    report = judge.evaluate_path(ROOT / "examples" / "observations" / "sample_rain_outside.json")
    skipped_ids = {s["element_id"] for s in report.skipped}
    assert "ODC.3.1.2.rain_intensity" in skipped_ids


def test_cli_evaluate():
    result = subprocess.run(
        [
            sys.executable, "-m", "odd_judge", "evaluate",
            "--project-root", str(ROOT),
            "--profile", "byd_tjp",
            "--obs", str(ROOT / "examples" / "observations" / "sample_curve_boundary.json"),
            "--switches", str(ROOT / "switches" / "default_switches.json"),
            "--format", "text",
        ],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    assert result.returncode == 1, result.stderr  # BOUNDARY → exit code 1 (odd_judge CLI)
    assert "BOUNDARY" in result.stdout


def test_cli_validate():
    result = subprocess.run(
        [sys.executable, "-m", "odd_judge", "validate", "--profile", str(ROOT / "profiles" / "byd_tjp.json")],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    assert result.returncode == 0, result.stderr
    assert "VALID" in result.stdout

"""MissingPolicy: per-element / per-tag fallback for missing observations."""

import json
from pathlib import Path

import pytest

from odd_judge.api import OddJudge
from odd_judge.config.loader import (
    load_profile_bundle,
    load_profile_with_extends,
    parse_profile_bundle,
)
from odd_judge.config.registry import clear as clear_registry
from odd_judge.config.schema import validate_profile_dict
from odd_judge.core.errors import ConfigError

ROOT = Path(__file__).resolve().parent.parent
SWITCHES = ROOT / "switches" / "default_switches.json"


def _element(element_id: str, tags=(), undefined_as=None, **extra) -> dict:
    el = {
        "element_id": element_id,
        "requirement": "allowed",
        "rule_type": "qualitative_bool",
        # dot-path key: dots inside the leaf name would be split by Observation.get
        "observation_key": f"scenery.{element_id.replace('.', '_')}",
        "tags": list(tags),
    }
    if undefined_as:
        el["undefined_as"] = undefined_as
    el.update(extra)
    return el


def _profile(elements, *, mode="restrictive", missing_policy=None) -> dict:
    data = {"profile_id": "test_policy", "definition_mode": mode, "elements": elements}
    if missing_policy is not None:
        data["missing_policy"] = missing_policy
    return data


def _judge(profile: dict) -> OddJudge:
    clear_registry()
    return OddJudge.from_dicts(profile)


def _states(judge: OddJudge, fields: dict) -> dict[str, str]:
    report = judge.evaluate({"fields": fields})
    return {r.element_id: r.state.value for r in report.results}


@pytest.fixture(autouse=True)
def _clean_registry():
    clear_registry()
    yield
    clear_registry()


def test_by_tag_policy_turns_missing_into_undefined():
    profile = _profile(
        [_element("ODC.T.1", tags=["scenery"])],
        missing_policy={"by_tag": {"scenery": "undefined"}},
    )
    assert _states(_judge(profile), {})["ODC.T.1"] == "UNDEFINED"


def test_missing_without_policy_keeps_restrictive_behaviour():
    profile = _profile([_element("ODC.T.1", tags=["scenery"])])
    assert _states(_judge(profile), {})["ODC.T.1"] == "OUTSIDE"


def test_undefined_policy_does_not_pull_overall_state_down():
    elements = [
        _element("ODC.T.1", tags=["scenery"]),
        _element("ODC.T.2", tags=["scenery"]),
    ]
    profile = _profile(elements, missing_policy={"by_tag": {"scenery": "undefined"}})
    judge = _judge(profile)
    report = judge.evaluate({"fields": {"scenery": {"ODC_T_2": True}}})
    assert report.state.value == "INSIDE"
    assert {u["element_id"] for u in report.undefined_handled} == {"ODC.T.1"}
    assert report.undefined_handled[0]["policy"] == "by_tag[scenery]"


def test_by_element_beats_by_tag():
    profile = _profile(
        [_element("ODC.T.1", tags=["scenery"])],
        missing_policy={
            "by_tag": {"scenery": "undefined"},
            "by_element": {"ODC.T.1": "not_allowed"},
        },
    )
    assert _states(_judge(profile), {})["ODC.T.1"] == "OUTSIDE"


def test_element_undefined_as_beats_policy():
    profile = _profile(
        [_element("ODC.T.1", tags=["scenery"], undefined_as="allowed")],
        missing_policy={"by_element": {"ODC.T.1": "not_allowed"}},
    )
    assert _states(_judge(profile), {})["ODC.T.1"] == "INSIDE"


def test_default_policy_applies_when_nothing_matches():
    profile = _profile(
        [_element("ODC.T.1", tags=["dynamic"])],
        missing_policy={"by_tag": {"scenery": "undefined"}, "default": "allowed"},
    )
    assert _states(_judge(profile), {})["ODC.T.1"] == "INSIDE"


def test_default_mode_falls_through_to_definition_mode():
    profile = _profile(
        [_element("ODC.T.1", tags=["dynamic"])],
        mode="permissive",
        missing_policy={"default": "mode"},
    )
    assert _states(_judge(profile), {})["ODC.T.1"] == "INSIDE"


def test_invalid_policy_value_is_rejected():
    profile = _profile([_element("ODC.T.1")], missing_policy={"default": "maybe"})
    errors = validate_profile_dict(profile)
    assert any("missing_policy.default" in e for e in errors)
    with pytest.raises(ConfigError, match="missing_policy"):
        _judge(profile)


def test_unknown_policy_key_is_rejected():
    profile = _profile([_element("ODC.T.1")], missing_policy={"by_source": {"map": "undefined"}})
    errors = validate_profile_dict(profile)
    assert any("unknown key" in e for e in errors)


def test_policy_merges_later_wins_across_extends_and_override(tmp_path):
    base = tmp_path / "base.json"
    ext = tmp_path / "ext.json"
    over = tmp_path / "over.json"
    base.write_text(
        json.dumps(
            _profile(
                [_element("ODC.T.1", tags=["scenery"])],
                missing_policy={"by_tag": {"scenery": "undefined"}, "default": "mode"},
            )
        ),
        encoding="utf-8",
    )
    ext.write_text(
        json.dumps(
            _profile(
                [_element("ODC.T.2", tags=["scenery"])],
                missing_policy={"by_tag": {"scenery": "not_allowed"}},
            )
        ),
        encoding="utf-8",
    )
    over.write_text(
        json.dumps(_profile([], missing_policy={"by_tag": {"scenery": "allowed"}})),
        encoding="utf-8",
    )
    bundle = load_profile_bundle(base, extends_paths=[ext], override_paths=[over])
    assert bundle.missing_policy is not None
    assert bundle.missing_policy.by_tag["scenery"] == "allowed"


def test_bundle_and_legacy_api_agree_on_rules(tmp_path):
    profile_path = tmp_path / "p.json"
    profile_path.write_text(json.dumps(_profile([_element("ODC.T.1")])), encoding="utf-8")
    bundle = load_profile_bundle(profile_path)
    pid, mode, rules = load_profile_with_extends(profile_path)
    assert bundle.profile_id == pid
    assert bundle.definition_mode == mode
    assert bundle.rules == rules


def test_switches_config_has_no_dead_keys():
    data = json.loads(SWITCHES.read_text(encoding="utf-8"))
    consumed = {
        "global_enabled",
        "by_element",
        "by_rule_type",
        "by_tag",
        "boundary_warnings_enabled",
    }
    assert set(data) <= consumed, f"unconsumed switch keys: {set(data) - consumed}"
    assert "on_switch_mismatch" not in data


def test_parse_profile_bundle_without_policy_is_none():
    bundle = parse_profile_bundle(_profile([_element("ODC.T.1")]))
    assert bundle.missing_policy is None

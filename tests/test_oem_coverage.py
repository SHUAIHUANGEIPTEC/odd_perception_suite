"""Drift guard: coverage/oem_cn_l2l3.json must stay consistent with profiles/byd_tjp.json.

The coverage file is documentation-grade data (which OEM does which ODD dimension),
not a judgment profile. These tests pin two invariants:

1. its own fields stay within the declared enums and summary counts match reality;
2. every ``profile_elements`` id still exists in the BYD TJP profile, and the
   "gap" and "has elements" fields never contradict each other.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COVERAGE = ROOT / "coverage" / "oem_cn_l2l3.json"
PROFILE = ROOT / "profiles" / "byd_tjp.json"
EXT_PROFILE = ROOT / "profiles" / "odc_ext_cn.json"
CITY_OVERRIDE = ROOT / "profiles" / "overrides" / "city_noa_cn.json"

VALID_TIERS = {"mandatory", "mainstream", "leading", "rare"}
VALID_THRESHOLD_SOURCES = {"regulation", "oem_declared"}
VALID_CONFIDENCE = {"high", "medium", "low"}


def _coverage() -> dict:
    return json.loads(COVERAGE.read_text(encoding="utf-8"))


def _dimensions() -> list[dict]:
    return _coverage()["dimensions"]


def test_dimensions_present_and_unique():
    dims = _dimensions()
    assert len(dims) == 17
    ids = [d["dim_id"] for d in dims]
    assert len(ids) == len(set(ids))


def test_enum_fields():
    data = _coverage()
    sources = set(data["source_enum"])
    layers = set(data["layer_enum"])
    tiers = set(data["tier_definitions"])
    for d in data["dimensions"]:
        name = d["dim_id"]
        assert d["tier"] in VALID_TIERS, name
        assert d["tier"] in tiers, name
        assert d["actual_source"] in sources, name
        assert set(d["layers"]) <= layers, name
        assert d["threshold_source"] in VALID_THRESHOLD_SOURCES, name
        assert d["confidence"] in VALID_CONFIDENCE, name


def test_profile_elements_exist_in_byd_tjp():
    profile = json.loads(PROFILE.read_text(encoding="utf-8"))
    known = {el["element_id"] for el in profile["elements"]}
    for d in _dimensions():
        for eid in d["profile_elements"]:
            assert eid in known, f"{d['dim_id']}: unknown element_id {eid}"


def test_extension_elements_exist_in_profiles():
    known = set()
    for path in (EXT_PROFILE, CITY_OVERRIDE):
        profile = json.loads(path.read_text(encoding="utf-8"))
        known |= {el["element_id"] for el in profile["elements"]}
    for d in _dimensions():
        for eid in d.get("extension_elements", []):
            assert eid in known, f"{d['dim_id']}: unknown extension element {eid}"


def test_gap_field_matches_elements():
    for d in _dimensions():
        covered = bool(d["profile_elements"]) or bool(d.get("extension_elements"))
        assert (d["gap_zh"] is None) == covered, d["dim_id"]


def test_summary_counts_match_dimensions():
    data = _coverage()
    dims = data["dimensions"]
    summary = data["summary_zh"]
    for layer in data["layer_enum"]:
        assert layer in summary, layer
        assert summary[layer] == sum(1 for d in dims if layer in d["layers"]), layer
    assert summary["stable_degradation"] <= summary["online_judged"]

"""Drift guard: io_contract_map.json must cover every profile observation_key.

The map records how each judge-consumed field is sourced from the perception
output dictionary (感知输入输出数据定义.xlsx) — see docs/io_contract_review.md.
Two invariants are pinned:

1. every ``observation_key`` used by any profile has a map entry
   (status covered / convention / gap; gaps must carry a ``requested`` field);
2. no orphan entries: every map key is consumed by at least one profile.
"""

import glob
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAP_PATH = ROOT / "odd_suite" / "mapping" / "tables" / "io_contract_map.json"
PROFILE_PATHS = [
    ROOT / "profiles" / "byd_tjp.json",
    ROOT / "profiles" / "odc_skeleton.json",
    ROOT / "profiles" / "odc_ext_cn.json",
] + sorted(glob.glob(str(ROOT / "profiles" / "overrides" / "*.json")))

VALID_STATUSES = {"covered", "convention", "gap"}


def _map() -> dict:
    return json.loads(MAP_PATH.read_text(encoding="utf-8"))["keys"]


def _profile_keys() -> set[str]:
    keys: set[str] = set()
    for path in PROFILE_PATHS:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        for el in data.get("elements", []):
            key = el.get("observation_key")
            if key:
                keys.add(key)
    return keys


def test_every_profile_key_has_map_entry():
    missing = sorted(_profile_keys() - set(_map()))
    assert not missing, f"observation_key(s) missing from io_contract_map.json: {missing}"


def test_no_orphan_map_entries():
    orphans = sorted(set(_map()) - _profile_keys())
    assert not orphans, f"io_contract_map.json entry(ies) not used by any profile: {orphans}"


def test_statuses_valid_and_gaps_have_request():
    keys = _map()
    bad_status = sorted(k for k, v in keys.items() if v.get("status") not in VALID_STATUSES)
    assert not bad_status, f"invalid status: {bad_status}"
    gaps_without_request = sorted(
        k for k, v in keys.items() if v["status"] == "gap" and not v.get("requested")
    )
    assert not gaps_without_request, f"gap entries without 'requested': {gaps_without_request}"


def test_covered_entries_declare_source():
    keys = _map()
    no_source = sorted(
        k for k, v in keys.items() if v["status"] == "covered" and not v.get("source")
    )
    assert not no_source, f"covered entries without 'source': {no_source}"

"""CLI smoke tests: demo / judge / validate / list-rules (no torch, no GPU)."""

import json

from odd_suite import cli


def test_demo_all_scenarios(capsys):
    code = cli.main(["demo", "--format", "json"])
    out = capsys.readouterr().out
    assert code == 0
    # three JSON documents back to back; each contains a judgment state
    states = []
    decoder = json.JSONDecoder()
    idx = 0
    text = out.strip()
    while idx < len(text):
        obj, end = decoder.raw_decode(text, idx)
        states.append(obj["judgment"]["state"])
        idx = end
        while idx < len(text) and text[idx] in " \n\r\t":
            idx += 1
    assert states == ["INSIDE", "BOUNDARY", "OUTSIDE"]


def test_judge_scenario(capsys):
    code = cli.main(["judge", "--scenario", "rain_outside", "--format", "json"])
    out = capsys.readouterr().out
    assert code == 0
    assert json.loads(out)["state"] == "OUTSIDE"


def test_judge_obs_file(capsys):
    code = cli.main(
        ["judge", "--obs", "examples/observations/sample_highway_inside.json", "--format", "json"]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert json.loads(out)["state"] == "INSIDE"


def test_validate_and_list(capsys):
    assert cli.main(["validate", "--profile", "byd_tjp"]) == 0
    out = capsys.readouterr().out
    assert "VALID" in out
    assert cli.main(["list-rules", "--profile", "byd_tjp", "--tag", "environment"]) == 0


def test_missing_profile_is_config_error(capsys):
    code = cli.main(["judge", "--scenario", "rain_outside", "--profile", "no_such_profile"])
    assert code == 2
    assert "error:" in capsys.readouterr().err


def test_run_with_detections_json(tmp_path, capsys):
    payload = {
        "results": {
            "tok-1": [
                {
                    "sample_token": "tok-1",
                    "translation": [0.0, 0.0, 0.0],
                    "size": [2.0, 4.5, 1.6],
                    "rotation": [1.0, 0.0, 0.0, 0.0],
                    "velocity": [0.0, 0.0],
                    "detection_name": "traffic_cone",
                    "detection_score": 0.8,
                }
            ]
        }
    }
    det_json = tmp_path / "pred.json"
    det_json.write_text(json.dumps(payload))
    sidecar = "examples/sidecar_highway.json"
    code = cli.main(
        [
            "run",
            "--no-attr",
            "--detections-json",
            str(det_json),
            "--sidecar",
            sidecar,
            "--format",
            "json",
        ]
    )
    out = capsys.readouterr().out
    assert code == 0
    report = json.loads(out)
    assert report["detections"]["count"] == 1
    assert report["observation"]["fields"]["dynamic"]["obstacles_over_0_3m"] == 1


def test_out_refuses_overwrite(tmp_path, capsys):
    target = tmp_path / "report.json"
    target.write_text("existing")
    code = cli.main(["judge", "--scenario", "rain_outside", "--out", str(target)])
    assert code == 2
    assert "Refusing to overwrite" in capsys.readouterr().err

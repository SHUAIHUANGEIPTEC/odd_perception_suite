"""Unified CLI: demo | attr | judge | run | validate | list-rules | weights-check."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from odd_judge.api import OddJudge
from odd_judge.config.loader import load_profile_with_extends
from odd_judge.config.schema import validate_profile_dict
from odd_judge.perception.mock import SCENARIOS, MockAdapter
from odd_judge.report.render import render as render_judgment
from odd_suite import __version__
from odd_suite.adapters.detect_adapter import Sparse4DDetector, parse_nuscenes_predictions
from odd_suite.adapters.mock_adapter import build_demo_scenario, list_demo_scenarios
from odd_suite.errors import ConfigError, MissingDependencyError, SuiteError
from odd_suite.paths import REPO_ROOT, profiles_dir, weights_dir
from odd_suite.pipeline.config import build_config, resolve_profile, resolve_switches
from odd_suite.pipeline.runner import Pipeline
from odd_suite.report.render import render_pipeline
from odd_suite.schema import Detection3D


def _write_or_print(text: str, out: str | None) -> None:
    if out:
        target = Path(out)
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite existing file: {out}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text + "\n", encoding="utf-8")
        print(f"Report written to {out}")
    else:
        print(text)


def _resolve_extra(paths: list[str] | None, subdir: str | None = None) -> list[Path]:
    resolved: list[Path] = []
    for raw in paths or []:
        candidate = Path(raw)
        if candidate.exists():
            resolved.append(candidate)
            continue
        named = profiles_dir() / subdir / raw if subdir else profiles_dir() / raw
        if named.exists():
            resolved.append(named)
            continue
        raise ConfigError(f"File not found: {raw}")
    return resolved


# ---------------------------------------------------------------- demo


def cmd_demo(args: argparse.Namespace) -> int:
    names = [args.scenario] if args.scenario else list_demo_scenarios()
    config = build_config(profile=args.profile, switches=args.switches)
    pipeline = Pipeline(config)
    blocks: list[str] = []
    for name in names:
        scenario = build_demo_scenario(name)
        result = pipeline.run(
            attr_override=scenario.attr,
            detections=scenario.detections,
            sidecar={"fields": scenario.sidecar_fields},
            timestamp=scenario.timestamp,
            source=f"demo:{name}",
        )
        if args.format == "json":
            blocks.append(render_pipeline(result, fmt="json"))
        else:
            blocks.append(f"\n===== DEMO SCENARIO: {name} =====")
            blocks.append(render_pipeline(result, fmt="text"))
    text = "\n".join(blocks)
    _write_or_print(text, args.out)
    return 0


# ---------------------------------------------------------------- attr


def cmd_attr(args: argparse.Namespace) -> int:
    from odd_suite.vendor.attr.checkpoint import AttrClassifier

    checkpoint = Path(args.checkpoint) if args.checkpoint else weights_dir() / "attr_best.pth"
    if not checkpoint.exists():
        raise ConfigError(f"Checkpoint not found: {checkpoint}")
    classifier = AttrClassifier(checkpoint, args.device, args.weather_threshold)
    records: list[dict[str, Any]] = []
    for image in args.images:
        result = classifier.predict(Path(image))
        records.append(
            {
                "image": result.image,
                "prediction": result.prediction,
                "probabilities": result.probabilities,
            }
        )
    payload = {
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_epoch": classifier.epoch,
        "weather_threshold": args.weather_threshold,
        "results": records,
    }
    if args.format == "json":
        text = json.dumps(payload, ensure_ascii=False, indent=2)
    else:
        lines = [f"checkpoint: {payload['checkpoint']} (epoch {payload['checkpoint_epoch']})"]
        for rec in records:
            lines.append(f"\n{rec['image']}")
            for task, value in rec["prediction"].items():
                lines.append(f"  {task:<16} {value}")
        text = "\n".join(lines)
    _write_or_print(text, args.out)
    return 0


# ---------------------------------------------------------------- judge


def cmd_judge(args: argparse.Namespace) -> int:
    judge = OddJudge.from_files(
        resolve_profile(args.profile),
        resolve_switches(args.switches),
        extends_paths=_resolve_extra(args.extends) or None,
        override_paths=_resolve_extra(args.override, subdir="overrides") or None,
    )
    if args.scenario:
        from odd_suite.adapters.mock_adapter import list_demo_scenarios, load_demo_observation

        if args.scenario in list_demo_scenarios():
            report = judge.evaluate(load_demo_observation(args.scenario))
        else:
            report = judge.evaluate(MockAdapter(args.scenario).get_observation())
    elif args.obs:
        report = judge.evaluate_path(Path(args.obs))
    else:
        raise ConfigError("Provide --obs FILE or --scenario NAME")
    _write_or_print(render_judgment(report, fmt=args.format), args.out)
    return 0


# ---------------------------------------------------------------- run


def _detections_from_args(args: argparse.Namespace) -> list[Detection3D] | None:
    if args.detections_json:
        grouped = parse_nuscenes_predictions(Path(args.detections_json))
        if args.sample_token:
            if args.sample_token not in grouped:
                raise ConfigError(f"sample-token not in prediction JSON: {args.sample_token}")
            return grouped[args.sample_token]
        merged = [det for boxes in grouped.values() for det in boxes]
        print(
            f"note: no --sample-token; merged {len(merged)} boxes from {len(grouped)} samples",
            file=sys.stderr,
        )
        return merged
    if args.detect_ckpt:
        data_root = args.data_root or None
        detector = Sparse4DDetector(Path(args.detect_ckpt), data_root=data_root)
        return detector.detect(sample_token=args.sample_token)
    return None


def cmd_run(args: argparse.Namespace) -> int:
    config = build_config(
        profile=args.profile,
        switches=args.switches,
        attr_checkpoint=args.checkpoint,
        detect_ckpt=args.detect_ckpt,
        device=args.device,
        weather_threshold=args.weather_threshold,
        score_threshold=args.score_threshold,
        sidecar=args.sidecar,
        enable_attr=not args.no_attr,
        enable_detect=bool(args.detect_ckpt),
    )
    detections = _detections_from_args(args)
    pipeline = Pipeline(config)
    images = [Path(p) for p in args.images] if args.images else None
    sidecar = Path(args.sidecar) if args.sidecar else None
    result = pipeline.run(
        images=images,
        detections=detections,
        sidecar=sidecar,
        source="odd-suite run",
    )
    _write_or_print(render_pipeline(result, fmt=args.format), args.out)
    return 0


# ---------------------------------------------------------------- validate / list-rules


def cmd_validate(args: argparse.Namespace) -> int:
    path = resolve_profile(args.profile)
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    errors = validate_profile_dict(data)
    if errors:
        print(f"INVALID: {path}")
        for err in errors:
            print(f"  - {err}")
        return 1
    print(f"VALID: {path}  ({len(data.get('elements', []))} elements)")
    return 0


def cmd_list_rules(args: argparse.Namespace) -> int:
    pid, mode, rules = load_profile_with_extends(resolve_profile(args.profile))
    if args.tag:
        rules = [r for r in rules if args.tag in r.tags]
    print(f"profile={pid}  mode={mode.value}  rules={len(rules)}")
    for rule in rules:
        tags = ",".join(rule.tags) if rule.tags else "-"
        codes = rule.warning_codes.get("outside") or rule.warning_codes.get("boundary") or ""
        print(
            f"  {rule.element_id:<38} {rule.rule_type:<18} "
            f"req={rule.requirement.value:<12} tags={tags:<24} code={codes}"
        )
    return 0


# ---------------------------------------------------------------- weights-check


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cmd_weights_check(args: argparse.Namespace) -> int:
    manifest_path = REPO_ROOT / "MANIFEST.json"
    with manifest_path.open(encoding="utf-8") as f:
        manifest = json.load(f)
    failed = False
    for entry in manifest.get("weights", []):
        path = REPO_ROOT / entry["path"]
        label = entry["id"]
        if entry.get("symlink"):
            if path.is_symlink() and not path.exists():
                target = path.resolve() if path.is_symlink() else "?"
                print(f"FAIL  {label}: broken symlink -> {target}")
                failed = True
                continue
            if not path.exists():
                print(f"FAIL  {label}: missing ({path})")
                failed = True
                continue
            print(f"ok    {label}: symlink alive -> {Path(entry['source']).name}")
        elif not path.exists():
            print(f"FAIL  {label}: missing ({path})")
            failed = True
            continue
        expected = entry.get("sha256")
        if expected and path.is_file():
            actual = _sha256(path)
            if actual != expected:
                print(f"FAIL  {label}: sha256 mismatch ({actual})")
                failed = True
                continue
            print(f"ok    {label}: sha256 matches")
    anchor = manifest.get("anchor", {}).get("path")
    if anchor:
        anchor_path = REPO_ROOT / anchor
        print(("ok    " if anchor_path.exists() else "FAIL  ") + f"anchor: {anchor}")
        failed = failed or not anchor_path.exists()
    return 1 if failed else 0


# ---------------------------------------------------------------- parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="odd-suite",
        description="Unified ODD perception suite: Sparse4D detection + ODD attribute "
        "classification + rule-based ODD judgment",
    )
    parser.add_argument("--version", action="version", version=f"odd-suite {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser, *, profile_default: bool = True) -> None:
        if profile_default:
            p.add_argument(
                "--profile", default=None, help="ODD profile name or path (default: byd_tjp)"
            )
        p.add_argument(
            "--switches", default=None, help="Switches JSON (default: default_switches)"
        )
        p.add_argument("--format", choices=["text", "json"], default="text")
        p.add_argument("--out", default=None, help="Write report to a new file")

    p_demo = sub.add_parser("demo", help="End-to-end mock pipeline (no GPU/torch needed)")
    add_common(p_demo)
    p_demo.add_argument("--scenario", default=None, help=f"{list_demo_scenarios()} or all")
    p_demo.set_defaults(func=cmd_demo)

    p_attr = sub.add_parser("attr", help="Run ODD attribute classification on images")
    p_attr.add_argument("images", nargs="+")
    p_attr.add_argument("--checkpoint", default=None, help="Default: weights/attr_best.pth")
    p_attr.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p_attr.add_argument("--weather-threshold", type=float, default=0.5)
    p_attr.add_argument("--format", choices=["text", "json"], default="json")
    p_attr.add_argument("--out", default=None)
    p_attr.set_defaults(func=cmd_attr)

    p_judge = sub.add_parser("judge", help="Evaluate an observation against an ODD profile")
    add_common(p_judge)
    p_judge.add_argument("--obs", default=None, help="Observation JSON path")
    p_judge.add_argument("--scenario", default=None, help=f"Mock scenario: {list(SCENARIOS)}")
    p_judge.add_argument("--extends", nargs="*", default=None, help="Extended profile paths")
    p_judge.add_argument("--override", nargs="*", default=None, help="Override profile paths")
    p_judge.set_defaults(func=cmd_judge)

    p_run = sub.add_parser("run", help="Full pipeline: attrs + detections + sidecar -> judgment")
    add_common(p_run)
    p_run.add_argument(
        "--images", nargs="*", default=None, help="Image(s) for attribute classifier"
    )
    p_run.add_argument(
        "--checkpoint", default=None, help="Attr checkpoint (default: weights/attr_best.pth)"
    )
    p_run.add_argument("--no-attr", action="store_true", help="Skip attribute classification")
    p_run.add_argument("--detections-json", default=None, help="Offline nuScenes prediction JSON")
    p_run.add_argument(
        "--detect-ckpt", default=None, help="Sparse4D checkpoint (runs vendor inference)"
    )
    p_run.add_argument(
        "--data-root", default=None, help="nuScenes root (default: NUSCENES_DATA_DIR)"
    )
    p_run.add_argument("--sample-token", default=None, help="Frame token to pick from detections")
    p_run.add_argument("--sidecar", default=None, help="Sidecar observation JSON (scenery/vehicle)")
    p_run.add_argument("--device", choices=["cpu", "cuda"], default=None)
    p_run.add_argument("--weather-threshold", type=float, default=None)
    p_run.add_argument("--score-threshold", type=float, default=None)
    p_run.set_defaults(func=cmd_run)

    p_val = sub.add_parser("validate", help="Validate a profile JSON")
    p_val.add_argument("--profile", default="byd_tjp")
    p_val.set_defaults(func=cmd_validate)

    p_list = sub.add_parser("list-rules", help="List rules in a profile")
    p_list.add_argument("--profile", default="byd_tjp")
    p_list.add_argument("--tag", default=None)
    p_list.set_defaults(func=cmd_list_rules)

    p_weights = sub.add_parser("weights-check", help="Verify weight checksums / symlink liveness")
    p_weights.set_defaults(func=cmd_weights_check)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except MissingDependencyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except (SuiteError, ConfigError, FileNotFoundError, FileExistsError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())

"""CLI: evaluate | validate | list-rules."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .api import OddJudge
from .config.loader import load_profile_with_extends, load_switches
from .config.schema import validate_profile_dict, validate_switches_dict
from .report.render import render


def _resolve_profile_path(name_or_path: str, base: Path) -> Path:
    p = Path(name_or_path)
    if p.exists():
        return p
    # try profiles/<name>.json
    cand = base / "profiles" / f"{name_or_path}.json"
    if cand.exists():
        return cand
    cand2 = base / "profiles" / name_or_path
    if cand2.exists():
        return cand2
    raise FileNotFoundError(f"Profile not found: {name_or_path}")


def cmd_evaluate(args: argparse.Namespace) -> int:
    base = Path(args.project_root) if args.project_root else Path.cwd()
    profile_path = _resolve_profile_path(args.profile, base)

    extends = []
    if args.extends:
        for e in args.extends:
            extends.append(_resolve_profile_path(e, base))
    overrides = []
    if args.override:
        for o in args.override:
            overrides.append(Path(o) if Path(o).exists() else base / "profiles" / "overrides" / o)

    switches_path = args.switches
    if switches_path and not Path(switches_path).exists():
        switches_path = base / args.switches

    judge = OddJudge.from_files(
        profile_path,
        switches_path=switches_path,
        extends_paths=extends or None,
        override_paths=overrides or None,
    )

    if args.scenario:
        from .perception.mock import MockAdapter

        adapter = MockAdapter(args.scenario)
        reports = [judge.evaluate(adapter.get_observation())]
    else:
        obs_path = Path(args.obs) if Path(args.obs).exists() else base / args.obs
        if args.stream:
            reports = judge.evaluate_stream(obs_path)
        else:
            reports = [judge.evaluate_path(obs_path)]

    fmt = args.format
    outputs = [render(r, fmt=fmt) for r in reports]
    text = "\n".join(outputs)

    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"Report written to {args.out}")
    else:
        print(text)

    # Exit code reflects worst state
    worst = max((r.state.value for r in reports), key=lambda s: {"OUTSIDE": 3, "BOUNDARY": 2, "INSIDE": 1, "UNDEFINED": 0, "SKIPPED": 0}.get(s, 0))
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    path = Path(args.profile)
    if not path.exists():
        base = Path(args.project_root) if args.project_root else Path.cwd()
        path = _resolve_profile_path(args.profile, base)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    errors = validate_profile_dict(data)
    if errors:
        print(f"INVALID: {path}")
        for e in errors:
            print(f"  - {e}")
        return 1
    print(f"VALID: {path}  ({len(data.get('elements', []))} elements)")
    return 0


def cmd_list_rules(args: argparse.Namespace) -> int:
    base = Path(args.project_root) if args.project_root else Path.cwd()
    profile_path = _resolve_profile_path(args.profile, base)
    pid, mode, rules = load_profile_with_extends(profile_path)
    if args.tag:
        rules = [r for r in rules if args.tag in r.tags]
    print(f"profile={pid}  mode={mode.value}  rules={len(rules)}")
    for r in rules:
        tags = ",".join(r.tags) if r.tags else "-"
        codes = r.warning_codes.get("outside") or r.warning_codes.get("boundary") or ""
        print(f"  {r.element_id:<40} {r.rule_type:<20} req={r.requirement.value:<12} tags={tags:<16} code={codes}")
    return 0


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--project-root", default=None, help="Project root for resolving relative paths")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="odd_judge", description="ODD Judgment Engine")
    _add_common(parser)
    sub = parser.add_subparsers(dest="command", required=True)

    p_eval = sub.add_parser("evaluate", help="Evaluate an observation against an ODD profile")
    _add_common(p_eval)
    p_eval.add_argument("--profile", default="byd_tjp", help="Profile name or path")
    p_eval.add_argument("--obs", default=None, help="Observation JSON/JSONL path")
    p_eval.add_argument("--scenario", default=None, help="Mock scenario name (e.g. highway_inside)")
    p_eval.add_argument("--switches", default=None, help="Switches JSON path")
    p_eval.add_argument("--extends", nargs="*", default=None, help="Extended profile paths")
    p_eval.add_argument("--override", nargs="*", default=None, help="Override profile paths")
    p_eval.add_argument("--format", choices=["json", "text"], default="text")
    p_eval.add_argument("--out", default=None, help="Write report to file")
    p_eval.add_argument("--stream", action="store_true", help="Treat obs as JSONL stream")
    p_eval.set_defaults(func=cmd_evaluate)

    p_val = sub.add_parser("validate", help="Validate a profile JSON")
    _add_common(p_val)
    p_val.add_argument("--profile", required=True)
    p_val.set_defaults(func=cmd_validate)

    p_list = sub.add_parser("list-rules", help="List rules in a profile")
    _add_common(p_list)
    p_list.add_argument("--profile", default="byd_tjp")
    p_list.add_argument("--tag", default=None)
    p_list.set_defaults(func=cmd_list_rules)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

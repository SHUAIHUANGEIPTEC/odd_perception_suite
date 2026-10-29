"""Generate the AUTO-GEN warning-code tables in docs/warning_codes.md.

Scans profiles/*.json and profiles/overrides/*.json, extracts every
``warning_codes`` entry and renders one markdown table per profile into the
``<!-- AUTO-GEN:BEGIN -->`` / ``<!-- AUTO-GEN:END -->`` marker region of
docs/warning_codes.md. Hand-written sections outside the markers are kept.

Usage:
    python scripts/gen_warning_codes_doc.py          # regenerate in place
    python scripts/gen_warning_codes_doc.py --check  # exit 1 if doc is stale
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROFILES_DIR = ROOT / "profiles"
DOC_PATH = ROOT / "docs" / "warning_codes.md"

BEGIN_MARK = "<!-- AUTO-GEN:BEGIN -->"
END_MARK = "<!-- AUTO-GEN:END -->"

# Fixed emission order so the tables do not depend on JSON key order.
STATE_ORDER = ("outside", "boundary")
STATE_ZH = {"outside": "OUTSIDE（域外）", "boundary": "BOUNDARY（边界带）"}


def _fmt_num(v: float) -> str:
    return f"{v:g}"


def _cell(v: object) -> str:
    return str(v).replace("|", "\\|") if v not in (None, "") else "—"


def _condition_text(el: dict) -> str:
    """Human-readable trigger condition for one element."""
    rt = el.get("rule_type", "")
    op = el.get("op", "")
    if rt == "numeric_range":
        if op == "between" and el.get("range"):
            lo, hi = el["range"]
            base = f"between [{_fmt_num(lo)}, {_fmt_num(hi)}]"
        elif el.get("threshold") is not None:
            base = f"{op} {_fmt_num(float(el['threshold']))}"
        else:
            base = op or "—"
        bm = el.get("boundary_margin") or {}
        if bm.get("value"):
            if bm.get("mode") == "relative":
                base += f"，边界带 ±{float(bm['value']) * 100:g}%"
            else:
                base += f"，边界带 ±{_fmt_num(float(bm['value']))}"
        return base
    if rt == "qualitative_enum":
        parts = []
        if el.get("allowed_values"):
            parts.append("∈ {" + ", ".join(str(v) for v in el["allowed_values"]) + "}")
        if el.get("forbidden_values"):
            parts.append("∉ {" + ", ".join(str(v) for v in el["forbidden_values"]) + "}")
        if el.get("boundary_values"):
            parts.append("边界值 {" + ", ".join(str(v) for v in el["boundary_values"]) + "}")
        return " ".join(parts) or "—"
    # qualitative_bool
    if el.get("expect") is not None:
        return f"expect={str(el['expect']).lower()}"
    return "—"


def _iter_profile_files() -> list[Path]:
    files = sorted(PROFILES_DIR.glob("*.json"))
    files += sorted((PROFILES_DIR / "overrides").glob("*.json"))
    return files


def build_block() -> str:
    """Render the markdown tables for every profile that defines warning codes."""
    lines: list[str] = []
    for path in _iter_profile_files():
        data = json.loads(path.read_text(encoding="utf-8"))
        rows: list[tuple[str, str, dict]] = []
        for el in data.get("elements", []):
            wc = el.get("warning_codes") or {}
            for state in STATE_ORDER:
                code = wc.get(state)
                if code:
                    rows.append((code, state, el))
        if not rows:
            continue
        rel = path.relative_to(ROOT)
        lines.append(f"### `{rel}` — profile_id: {data.get('profile_id', '?')}")
        lines.append("")
        lines.append("| 告警码 | 触发状态 | element_id | label_zh | 触发条件 | message_zh |")
        lines.append("|---|---|---|---|---|---|")
        for code, state, el in rows:
            cells = [
                f"`{code}`",
                STATE_ZH[state],
                f"`{_cell(el.get('element_id'))}`",
                _cell(el.get("label_zh")),
                _cell(_condition_text(el)),
                _cell(el.get("message_zh")),
            ]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def splice(doc: str, block: str) -> str:
    if BEGIN_MARK not in doc or END_MARK not in doc:
        raise SystemExit(f"error: markers {BEGIN_MARK!r} / {END_MARK!r} not found in {DOC_PATH}")
    head, rest = doc.split(BEGIN_MARK, 1)
    _, tail = rest.split(END_MARK, 1)
    return head + BEGIN_MARK + "\n" + block + END_MARK + tail


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify the doc is up to date, do not write")
    args = parser.parse_args(argv)

    if not DOC_PATH.exists():
        raise SystemExit(f"error: {DOC_PATH} not found")
    doc = DOC_PATH.read_text(encoding="utf-8")
    updated = splice(doc, build_block())

    if args.check:
        if updated != doc:
            print("docs/warning_codes.md is stale; run: python scripts/gen_warning_codes_doc.py")
            return 1
        print("docs/warning_codes.md is up to date")
        return 0

    if updated != doc:
        DOC_PATH.write_text(updated, encoding="utf-8")
        print(f"updated {DOC_PATH.relative_to(ROOT)}")
    else:
        print("no changes")
    return 0


if __name__ == "__main__":
    sys.exit(main())

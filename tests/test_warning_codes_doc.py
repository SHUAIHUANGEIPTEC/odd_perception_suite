"""Guard: docs/warning_codes.md code tables must match profiles/ (no drift)."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_warning_codes_doc_up_to_date():
    script = ROOT / "scripts" / "gen_warning_codes_doc.py"
    res = subprocess.run(
        [sys.executable, str(script), "--check"],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, res.stdout + res.stderr

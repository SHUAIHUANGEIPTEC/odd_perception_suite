"""Verify weight checksums and symlink liveness against MANIFEST.json."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from odd_suite.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(["weights-check"]))

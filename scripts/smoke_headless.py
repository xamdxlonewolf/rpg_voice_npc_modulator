# Copyright (C) 2026 Michael Cobb
# SPDX-License-Identifier: GPL-3.0-or-later

"""Launch Voice of the Realm and exit. Used by the bundle smoke test."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def frozen_binary() -> Path:
    if sys.platform == "win32":
        return ROOT / "dist" / "VoiceOfTheRealm" / "VoiceOfTheRealm.exe"
    return ROOT / "dist" / "VoiceOfTheRealm" / "VoiceOfTheRealm"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--frozen",
        action="store_true",
        help="Run dist/VoiceOfTheRealm instead of python -m votr",
    )
    args = parser.parse_args(argv)
    if args.frozen:
        binary = frozen_binary()
        if not binary.is_file():
            print(f"frozen binary missing: {binary}", file=sys.stderr)
            return 2
        cmd = [str(binary), "--headless"]
    else:
        cmd = [sys.executable, "-m", "votr", "--headless"]
    print(" ".join(cmd))
    return subprocess.call(cmd, cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())

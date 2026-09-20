# Copyright (C) 2026 Michael Cobb
# SPDX-License-Identifier: GPL-3.0-or-later

"""Write installer/version.iss from pyproject.toml."""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
from read_version import ROOT, read_version  # noqa: E402

OUT = ROOT / "installer" / "version.iss"


def main() -> int:
    version = read_version()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(f'#define MyAppVersion "{version}"\n', encoding="utf-8")
    print(OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

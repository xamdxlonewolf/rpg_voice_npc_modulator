# Copyright (C) 2026 Michael Cobb
# SPDX-License-Identifier: GPL-3.0-or-later

"""Read the project version from pyproject.toml (no extra deps)."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_version(pyproject: Path | None = None) -> str:
    path = pyproject or (ROOT / "pyproject.toml")
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("version") and "=" in stripped:
            return stripped.split("=", 1)[1].strip().strip("\"'")
    raise ValueError(f"version not found in {path}")


if __name__ == "__main__":
    print(read_version())

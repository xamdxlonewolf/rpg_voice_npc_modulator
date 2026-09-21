# Copyright (C) 2026 Michael Cobb
# SPDX-License-Identifier: GPL-3.0-or-later

"""Repo or frozen-bundle file locations."""

from __future__ import annotations

import sys
from pathlib import Path


def bundle_or_repo_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]


def _search_roots() -> list[Path]:
    roots = [bundle_or_repo_root(), Path(__file__).resolve().parent]
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        roots.append(Path(meipass))
    return roots


def notices_path() -> Path | None:
    for base in _search_roots():
        notice = base / "THIRD_PARTY_NOTICES.md"
        if notice.is_file():
            return notice
    return None


def license_path() -> Path | None:
    for base in _search_roots():
        path = base / "LICENSE"
        if path.is_file():
            return path
    return None

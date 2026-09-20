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


def notices_path() -> Path | None:
    names = ("THIRD_PARTY_NOTICES.md", "LICENSE")
    root = bundle_or_repo_root()
    here = Path(__file__).resolve().parent
    for base in (root, here, here.parent):
        notice = base / names[0]
        if notice.is_file():
            return notice
    return None


def license_path() -> Path | None:
    root = bundle_or_repo_root()
    path = root / "LICENSE"
    return path if path.is_file() else None

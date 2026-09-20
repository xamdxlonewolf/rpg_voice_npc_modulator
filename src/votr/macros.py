# Copyright (C) 2026 Michael Cobb
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tone Tags → slider Macros bundled as macros.json."""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib.resources import files
from typing import Any


@dataclass(frozen=True)
class Macro:
    tag: str
    description: str
    params: dict[str, Any]


def load_macros() -> dict[str, Macro]:
    raw = files("votr.assets").joinpath("macros.json").read_text(encoding="utf-8")
    data = json.loads(raw)
    macros: dict[str, Macro] = {}
    for tag, body in data.items():
        macros[tag] = Macro(
            tag=tag,
            description=str(body.get("description") or ""),
            params=dict(body.get("params") or {}),
        )
    return macros


def apply_macros(
    params: dict[str, Any], tags: list[str], macros: dict[str, Macro]
) -> dict[str, Any]:
    """Patch params in tag order. Later tags win on overlapping keys."""
    merged = dict(params)
    for tag in tags:
        macro = macros.get(tag)
        if macro is None:
            continue
        merged.update(macro.params)
    return merged

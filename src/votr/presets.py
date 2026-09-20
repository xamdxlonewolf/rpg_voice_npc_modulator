# Copyright (C) 2026 Michael Cobb
# SPDX-License-Identifier: GPL-3.0-or-later

"""Voice presets: named slider recipes plus Tone Tags, bundled as presets.json.

A preset is a starting point. Using one creates a fresh draft Voice the GM can
Preview and tune; it is not a link, so later edits never change the preset.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from importlib.resources import files

from votr.voice import DSP_ENGINE_ID, Voice


@dataclass(frozen=True)
class Preset:
    name: str
    description: str
    tone_tags: tuple[str, ...] = ()
    params: dict[str, float] = field(default_factory=dict)
    colour: str = "#5c4d7a"
    engine_id: str = DSP_ENGINE_ID


def load_presets() -> list[Preset]:
    raw = files("votr.assets").joinpath("presets.json").read_text(encoding="utf-8")
    data = json.loads(raw)
    presets: list[Preset] = []
    for body in data:
        presets.append(
            Preset(
                name=str(body["name"]),
                description=str(body.get("description") or ""),
                tone_tags=tuple(str(tag) for tag in body.get("tone_tags") or ()),
                params={k: float(v) for k, v in (body.get("params") or {}).items()},
                colour=str(body.get("colour") or "#5c4d7a"),
                engine_id=str(body.get("engine_id") or DSP_ENGINE_ID),
            )
        )
    return presets


def preset_by_name(presets: list[Preset], name: str) -> Preset | None:
    wanted = name.strip().lower()
    return next((item for item in presets if item.name.lower() == wanted), None)


def voice_from_preset(preset: Preset) -> Voice:
    voice = Voice.new(preset.name)
    voice.engine_id = preset.engine_id
    voice.tone_hints = preset.description
    voice.tone_tags = list(preset.tone_tags)
    voice.params = dict(preset.params)
    voice.colour = preset.colour
    return voice

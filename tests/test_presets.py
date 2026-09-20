# Copyright (C) 2026 Michael Cobb
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

from pathlib import Path

import pytest

from votr.dsp import SCHEMA
from votr.macros import load_macros
from votr.presets import load_presets, preset_by_name, voice_from_preset
from votr.session import Session
from votr.spikes.pitch_core import stretch_available
from votr.voice import DSP_ENGINE_ID

_SPECS = {spec.key: spec for spec in SCHEMA}


def test_presets_load_and_stay_inside_the_schema() -> None:
    presets = load_presets()
    assert len(presets) >= 10
    names = [preset.name for preset in presets]
    assert len(set(names)) == len(names)
    macros = load_macros()
    for preset in presets:
        assert preset.description
        assert preset.engine_id == DSP_ENGINE_ID
        assert preset.tone_tags, preset.name
        for tag in preset.tone_tags:
            assert tag in macros, (preset.name, tag)
        assert preset.params, preset.name
        for key, value in preset.params.items():
            spec = _SPECS[key]
            assert spec.minimum <= value <= spec.maximum, (preset.name, key)


def test_presets_are_distinct_starting_points() -> None:
    presets = load_presets()
    signatures = {tuple(sorted(preset.params.items())) for preset in presets}
    assert len(signatures) == len(presets)


def test_voice_from_preset_is_a_fresh_copy() -> None:
    presets = load_presets()
    preset = preset_by_name(presets, "orc warchief")
    assert preset is not None
    voice = voice_from_preset(preset)
    assert voice.name == "Orc Warchief"
    assert voice.tone_tags == list(preset.tone_tags)
    assert voice.params == preset.params
    assert voice.tone_hints == preset.description
    voice.params["growl"] = 0.0
    assert preset.params["growl"] != 0.0
    assert preset_by_name(presets, "no such preset") is None


@pytest.mark.skipif(not stretch_available(), reason="DSP Engine needs python-stretch")
def test_session_edit_from_preset_sets_engine_and_is_unsaved(tmp_path: Path) -> None:
    session = Session(tmp_path)
    voice = session.edit_from_preset("Cave Troll")
    assert voice is not None
    assert session.draft.name == "Cave Troll"
    assert session.is_dirty()
    params = session.engine.params()
    assert params["pitch_semitones"] == pytest.approx(-7.0)
    assert params["formant_semitones"] == pytest.approx(-6.0)
    assert session.store.load_all() == []
    assert session.edit_from_preset("Nobody") is None

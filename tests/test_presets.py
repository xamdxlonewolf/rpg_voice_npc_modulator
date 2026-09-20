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
    assert params["pitch_semitones"] == pytest.approx(-6.0)
    assert params["formant_semitones"] == pytest.approx(-4.0)
    assert session.store.load_all() == []
    assert session.edit_from_preset("Nobody") is None


@pytest.mark.skipif(not stretch_available(), reason="DSP Engine needs python-stretch")
def test_saved_preset_edits_persist_and_win_over_the_bundle(tmp_path: Path) -> None:
    session = Session(tmp_path)
    session.edit_from_preset("Pixie")
    bundled = dict(session.draft.params)
    session.draft.params["pitch_semitones"] = 2.0
    session.draft.params["breath"] = 0.4
    saved = session.save_draft()
    assert saved.preset == "Pixie"

    # Reload from disk: the user JSON has the edit.
    reopened = Session(tmp_path)
    voice = reopened.voice_by_id(saved.id)
    assert voice is not None
    assert voice.params["pitch_semitones"] == pytest.approx(2.0)
    assert voice.params["breath"] == pytest.approx(0.4)

    # Picking the preset again opens the saved Voice, not the bundled recipe.
    again = reopened.edit_from_preset("Pixie")
    assert again is not None and again.id == saved.id
    assert reopened.draft.params["pitch_semitones"] == pytest.approx(2.0)
    assert reopened.engine.params()["pitch_semitones"] == pytest.approx(2.0)
    assert not reopened.is_dirty()

    # Switching Voices and coming back keeps it too.
    other = reopened.edit_from_preset("Lich", fresh=True)
    assert other is not None and other.name == "Lich"
    reopened.edit(voice)
    assert reopened.draft.params["pitch_semitones"] == pytest.approx(2.0)

    # A fresh copy is the bundled recipe under a non-colliding name.
    fresh = reopened.edit_from_preset("Pixie", fresh=True)
    assert fresh is not None
    assert fresh.name == "Pixie 2"
    assert fresh.params == bundled
    assert reopened.is_dirty()
    assert len(reopened.store.load_all()) == 1


def test_voice_json_without_preset_field_still_loads() -> None:
    from votr.voice import Voice

    voice = Voice.from_dict({"id": "x", "name": "Old"})
    assert voice.preset == ""
    assert Voice.from_dict(voice.to_dict()).preset == ""


def test_softened_presets_stay_in_character_but_not_slammed() -> None:
    presets = {preset.name: preset for preset in load_presets()}
    for name in (
        "Orc Warchief",
        "Cave Troll",
        "Elder Dragon",
        "Lich",
        "Through a Helmet",
    ):
        params = presets[name].params
        assert params.get("growl", 0.0) <= 0.4, name
        assert params.get("hollow", 0.0) <= 0.4, name
        assert params.get("room", 0.0) <= 0.3, name
        assert abs(params.get("formant_semitones", 0.0)) <= 5.0, name
        assert abs(params.get("pitch_semitones", 0.0)) <= 7.0, name
        # Still clearly that character, not dry.
        assert any(abs(value) >= 0.15 for value in params.values()), name
    assert presets["Elder Dragon"].params["pitch_semitones"] <= -6.0
    assert presets["Orc Warchief"].params["growl"] >= 0.3

# Copyright (C) 2026 Michael Cobb
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

from pathlib import Path

import pytest

from votr.dsp import SCHEMA
from votr.macros import load_macros
from votr.neural import NeuralDesigner
from votr.session import Session
from votr.spikes.pitch_core import stretch_available
from votr.voicedesign import (
    ACCENT_NOTE,
    IDENTITY_NOTE,
    LexiconDesigner,
    design_voice,
    designers,
)

_SPECS = {spec.key: spec for spec in SCHEMA}


def _in_schema(params: dict[str, float]) -> None:
    for key, value in params.items():
        spec = _SPECS[key]
        assert spec.minimum <= value <= spec.maximum, key


def test_prompt_picks_tags_and_nudges_sliders() -> None:
    design = LexiconDesigner().design("a gruff old dwarf blacksmith with a deep voice")
    assert "gravelly" in design.tone_tags
    assert "frail" in design.tone_tags
    macros = load_macros()
    baseline = macros["frail"].params["pitch_semitones"]
    # "deep" nudges pitch below what the last tag alone would set.
    assert design.params["pitch_semitones"] < baseline
    assert design.params["growl"] > 0.0
    assert design.name == "Gruff Old Dwarf"
    assert "deep" in design.matched
    assert design.designer == "lexicon"
    _in_schema(design.params)


def test_intensifiers_scale_nudges_and_values_are_clipped() -> None:
    mild = LexiconDesigner().design("a slightly deep voice")
    strong = LexiconDesigner().design("a very deep voice")
    plain = LexiconDesigner().design("a deep voice")
    assert strong.params["pitch_semitones"] < plain.params["pitch_semitones"]
    assert mild.params["pitch_semitones"] > plain.params["pitch_semitones"]
    huge = LexiconDesigner().design("very deep very low bass giant dragon troll demon")
    assert huge.params["pitch_semitones"] == pytest.approx(-12.0)
    _in_schema(huge.params)


def test_explicit_semitones_win() -> None:
    design = LexiconDesigner().design("a tiny pixie, pitch -5 semitones")
    assert "tiny" in design.tone_tags
    assert design.params["pitch_semitones"] == pytest.approx(-5.0)


def test_space_words_move_room_and_distance() -> None:
    design = LexiconDesigner().design("a ghost echoing far away in a cathedral")
    assert "ghostly" in design.tone_tags
    macros = load_macros()
    assert design.params["room"] > macros["ghostly"].params["room"]
    assert design.params["distance"] > macros["ghostly"].params["distance"]


def test_accent_and_identity_requests_are_noted_not_faked() -> None:
    accent = LexiconDesigner().design("a cheerful Irish innkeeper")
    assert ACCENT_NOTE in accent.notes
    defaults = {spec.key: spec.default for spec in SCHEMA}
    # "cheerful", "Irish", "innkeeper" carry no sound words: sliders stay dry.
    assert accent.params == defaults
    identity = LexiconDesigner().design("sound like a famous actor, deep")
    assert IDENTITY_NOTE in identity.notes
    assert identity.params["pitch_semitones"] < 0.0


def test_unrecognised_prompt_explains_itself() -> None:
    design = LexiconDesigner().design("Bartholomew the accountant")
    assert design.tone_tags == []
    assert any("No sound words" in note for note in design.notes)
    assert design.name == "Bartholomew Accountant"


def test_design_voice_falls_back_to_lexicon_when_neural_is_off() -> None:
    chain = designers()
    assert isinstance(chain[0], NeuralDesigner)
    ok, reason = chain[0].available()
    assert ok is False
    assert "not installed" in reason
    with pytest.raises(RuntimeError):
        chain[0].design("anything")
    design = design_voice("a whispering assassin")
    assert design.designer == "lexicon"
    assert "whisper" in design.tone_tags


@pytest.mark.skipif(not stretch_available(), reason="DSP Engine needs python-stretch")
def test_session_design_from_hints_fills_the_draft(tmp_path: Path) -> None:
    session = Session(tmp_path)
    design = session.design_from_hints("a booming giant in a great hall")
    assert session.draft.name == design.name == "Booming Giant Great"
    assert "giant" in session.draft.tone_tags
    assert session.draft.tone_hints == "a booming giant in a great hall"
    params = session.engine.params()
    assert params["pitch_semitones"] == pytest.approx(design.params["pitch_semitones"])
    assert params["room"] == pytest.approx(design.params["room"])
    session.draft.name = "Grom"
    session.design_from_hints("a tiny sprite")
    assert session.draft.name == "Grom"
    assert session.draft.tone_tags == ["tiny"]

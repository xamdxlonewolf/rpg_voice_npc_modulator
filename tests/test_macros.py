# Copyright (C) 2026 Michael Cobb
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

from votr.macros import apply_macros, load_macros


def test_macros_include_required_tags() -> None:
    macros = load_macros()
    expected = {
        "gravelly",
        "booming",
        "frail",
        "hollow",
        "tiny",
        "giant",
        "ghostly",
        "robotic",
        "nasal",
        "whisper",
        "regal",
        "sly",
    }
    assert expected <= set(macros)
    assert macros["gravelly"].description


def test_apply_order_later_tags_win() -> None:
    macros = load_macros()
    first = apply_macros({}, ["gravelly"], macros)
    both = apply_macros({}, ["gravelly", "booming"], macros)
    assert both["growl"] == macros["booming"].params["growl"]
    assert both["pitch_semitones"] == macros["booming"].params["pitch_semitones"]
    assert first["growl"] == macros["gravelly"].params["growl"]
    # gravelly-only keys remain unless booming overwrites them
    gravelly_only = "breath" in macros["gravelly"].params
    booming_has = "breath" in macros["booming"].params
    if gravelly_only and not booming_has:
        assert both["breath"] == macros["gravelly"].params["breath"]


def test_apply_macro_is_idempotent() -> None:
    macros = load_macros()
    once = apply_macros({"room": 0.9}, ["tiny"], macros)
    twice = apply_macros(once, ["tiny"], macros)
    assert once == twice

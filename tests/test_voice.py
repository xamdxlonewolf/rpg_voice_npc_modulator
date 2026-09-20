# Copyright (C) 2026 Michael Cobb
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import json
from pathlib import Path

from votr.store import VoiceStore
from votr.voice import Voice


def test_voice_round_trip_preserves_unknown_fields(tmp_path: Path) -> None:
    voice = Voice.new("Grimjaw")
    voice.tone_hints = "gravelly orc captain"
    voice.tone_tags = ["gravelly", "booming"]
    voice.params = {"pitch_semitones": -4.0}
    voice.extra["future_field"] = {"keep": True}
    store = VoiceStore(tmp_path)
    store.save(voice)
    loaded = store.load_all()
    assert len(loaded) == 1
    again = loaded[0]
    assert again.name == "Grimjaw"
    assert again.tone_hints == "gravelly orc captain"
    assert again.tone_tags == ["gravelly", "booming"]
    assert again.params["pitch_semitones"] == -4.0
    assert again.extra["future_field"] == {"keep": True}
    raw = json.loads(store.path_for(voice.id).read_text(encoding="utf-8"))
    assert raw["future_field"] == {"keep": True}


def test_corrupt_voice_file_is_skipped(tmp_path: Path) -> None:
    store = VoiceStore(tmp_path)
    bad = store.root / "broken.json"
    bad.write_text("{not-json", encoding="utf-8")
    good = Voice.new("OK")
    store.save(good)
    voices = store.load_all()
    assert [voice.name for voice in voices] == ["OK"]
    assert store.warnings

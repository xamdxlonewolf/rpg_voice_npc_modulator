# Copyright (C) 2026 Michael Cobb
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from votr.clips import (
    CLIP_SAMPLE_RATE,
    MAX_CLIP_SECONDS,
    ORIGIN_PLAYBACK,
    REFERENCE_CLIP_ID_KEY,
    ClipError,
    ClipLibrary,
    decode_audio,
)
from votr.neural_engine import NEURAL_ENGINE_ID, REFERENCE_CLIP_KEY
from votr.session import Session
from votr.spikes.pitch_core import stretch_available
from votr.voice import Voice
from votr.wavutil import read_wav, write_wav


def _speech(
    seconds: float, rate: int = CLIP_SAMPLE_RATE, hz: float = 180.0
) -> np.ndarray:
    t = np.arange(int(rate * seconds)) / rate
    return (0.3 * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def test_add_list_rename_delete_round_trip(tmp_path: Path) -> None:
    library = ClipLibrary(tmp_path)
    assert library.list() == []
    clip = library.add(_speech(6.0), CLIP_SAMPLE_RATE, "  Grumpy dwarf ")
    assert clip.name == "Grumpy dwarf"
    assert clip.seconds == pytest.approx(6.0, abs=0.25)
    assert clip.origin == "recorded" and clip.created
    assert library.path_for(clip.id).is_file()
    samples, rate = library.load(clip.id)
    assert rate == CLIP_SAMPLE_RATE and samples.size == int(clip.seconds * rate)

    # Fresh library instance reads the same metadata back from disk.
    again = ClipLibrary(tmp_path)
    assert [c.id for c in again.list()] == [clip.id]
    assert again.get(clip.id).name == "Grumpy dwarf"
    assert again.by_name("grumpy DWARF").id == clip.id

    second = library.add(_speech(3.0), CLIP_SAMPLE_RATE, "Grumpy dwarf")
    assert second.name == "Grumpy dwarf 2"
    assert library.rename(second.id, "Elf queen").name == "Elf queen"
    assert library.rename(second.id, "Grumpy dwarf").name == "Grumpy dwarf 2"
    assert [c.name for c in library.list()] == ["Grumpy dwarf", "Grumpy dwarf 2"]
    assert library.rename("nope", "x") is None

    library.delete(clip.id)
    assert library.get(clip.id) is None
    assert not library.path_for(clip.id).exists()
    assert library.load(clip.id) is None
    library.delete(clip.id)  # idempotent


def test_add_resamples_trims_and_caps_at_thirty_seconds(tmp_path: Path) -> None:
    library = ClipLibrary(tmp_path)
    padded = np.concatenate(
        [
            np.zeros(16_000, np.float32),
            _speech(2.0, rate=16_000),
            np.zeros(16_000, np.float32),
        ]
    )
    clip = library.add(padded, 16_000, "padded")
    assert 2.0 <= clip.seconds <= 2.3  # leading/trailing silence trimmed
    samples, rate = library.load(clip.id)
    assert rate == CLIP_SAMPLE_RATE
    long = library.add(_speech(45.0), CLIP_SAMPLE_RATE, "too long")
    assert long.seconds == pytest.approx(MAX_CLIP_SECONDS, abs=0.01)
    loud = library.add(_speech(2.0) * 4.0, CLIP_SAMPLE_RATE, "hot")
    assert float(np.max(np.abs(library.load(loud.id)[0]))) <= 0.99


def test_too_short_or_silent_clips_are_rejected(tmp_path: Path) -> None:
    library = ClipLibrary(tmp_path)
    with pytest.raises(ClipError, match="at least"):
        library.add(_speech(0.4), CLIP_SAMPLE_RATE, "blip")
    with pytest.raises(ClipError):
        library.add(
            np.zeros(CLIP_SAMPLE_RATE * 5, np.float32), CLIP_SAMPLE_RATE, "nothing"
        )
    assert library.list() == []


def test_import_wav_flac_and_reject_unknown(tmp_path: Path) -> None:
    library = ClipLibrary(tmp_path)
    wav = tmp_path / "voice.wav"
    write_wav(wav, _speech(4.0, rate=24_000), 24_000)
    clip = library.import_file(wav)
    assert clip.name == "voice" and clip.origin == "file:voice.wav"
    assert clip.seconds == pytest.approx(4.0, abs=0.2)
    assert library.load(clip.id)[1] == CLIP_SAMPLE_RATE

    from pedalboard.io import AudioFile

    flac = tmp_path / "elf.flac"
    with AudioFile(str(flac), "w", samplerate=44_100, num_channels=2) as out:
        stereo = np.stack(
            [_speech(3.0, rate=44_100), _speech(3.0, rate=44_100, hz=220)]
        )
        out.write(stereo)
    imported = library.import_file(flac, "Elf")
    assert imported.name == "Elf"
    assert imported.seconds == pytest.approx(3.0, abs=0.2)
    samples, rate = decode_audio(flac)
    assert rate == CLIP_SAMPLE_RATE and samples.ndim == 1

    with pytest.raises(ClipError, match="not a supported"):
        library.import_file(tmp_path / "notes.txt")
    (tmp_path / "broken.mp3").write_bytes(b"not audio")
    with pytest.raises(ClipError, match="Could not read"):
        library.import_file(tmp_path / "broken.mp3")


def test_corrupt_metadata_is_skipped_not_fatal(tmp_path: Path) -> None:
    library = ClipLibrary(tmp_path)
    good = library.add(_speech(2.0), CLIP_SAMPLE_RATE, "good")
    (library.root / "bad.json").write_text("{not json")
    (library.root / "orphan.json").write_text(json.dumps({"id": "orphan", "name": "o"}))
    assert [c.id for c in library.list()] == [good.id]


def test_voice_json_keeps_clip_id_and_path() -> None:
    voice = Voice.new("Mimic")
    voice.engine_id = NEURAL_ENGINE_ID
    voice.params = {
        REFERENCE_CLIP_ID_KEY: "abc",
        REFERENCE_CLIP_KEY: "/x/abc.wav",
        "mix": 1.0,
    }
    reloaded = Voice.from_dict(json.loads(json.dumps(voice.to_dict())))
    assert reloaded.engine_id == NEURAL_ENGINE_ID
    assert reloaded.params[REFERENCE_CLIP_ID_KEY] == "abc"
    assert reloaded.params[REFERENCE_CLIP_KEY] == "/x/abc.wav"


@pytest.mark.skipif(not stretch_available(), reason="DSP Engine needs python-stretch")
def test_session_selects_saves_and_reloads_a_mimic_clip(tmp_path: Path) -> None:
    session = Session(tmp_path)
    clip = session.clips.add(_speech(5.0), CLIP_SAMPLE_RATE, "Old wizard")
    session.edit_new()
    session.draft.name = "Wizard"
    session.set_engine_kind(NEURAL_ENGINE_ID)
    assert session.draft.engine_id == NEURAL_ENGINE_ID
    assert session.draft.params["mix"] == 1.0
    assert session.draft.params["quality"] == 1.0
    assert session.use_clip(clip.id) is not None
    assert session.draft_clip().id == clip.id
    assert session.draft.params[REFERENCE_CLIP_KEY] == str(
        session.clips.path_for(clip.id)
    )
    saved = session.save_draft()

    reopened = Session(tmp_path)
    voice = reopened.voice_by_id(saved.id)
    assert voice is not None and voice.engine_id == NEURAL_ENGINE_ID
    assert voice.params[REFERENCE_CLIP_ID_KEY] == clip.id
    resolved = reopened.resolved_params(voice.params)
    assert Path(resolved[REFERENCE_CLIP_KEY]).is_file()
    # A stale path is healed from the id.
    voice.params[REFERENCE_CLIP_KEY] = "C:/old/laptop/abc.wav"
    assert reopened.resolved_params(voice.params)[REFERENCE_CLIP_KEY].endswith(
        f"{clip.id}.wav"
    )
    # Library works and the Voice saves even though the Neural Engine is not ready here.
    assert not reopened.engine_installed(NEURAL_ENGINE_ID)
    reopened.edit(voice)
    assert reopened.preview_engine() is reopened.engine
    assert session.use_clip("missing") is None
    session.clear_clip()
    assert session.draft_clip() is None
    session.set_engine_kind("bogus")
    assert session.draft.engine_id == NEURAL_ENGINE_ID


def test_read_wav_of_saved_clip_matches_added_audio(tmp_path: Path) -> None:
    library = ClipLibrary(tmp_path)
    audio = _speech(2.0)
    clip = library.add(audio, CLIP_SAMPLE_RATE, "check")
    samples, _ = read_wav(library.path_for(clip.id))
    n = min(samples.size, audio.size)
    assert float(np.corrcoef(samples[:n], audio[:n])[0, 1]) > 0.999


def test_clips_are_stored_at_a_healthy_level_and_remember_source_peak(
    tmp_path: Path,
) -> None:
    library = ClipLibrary(tmp_path)
    quiet = library.add(_speech(3.0) * 0.02, CLIP_SAMPLE_RATE, "quiet mic")
    assert quiet.source_peak_db == pytest.approx(-44.4, abs=0.5)
    assert quiet.was_quiet and "not boosted" in quiet.level_note
    samples, _ = library.load(quiet.id)
    assert float(np.max(np.abs(samples))) == pytest.approx(0.006, abs=0.001)
    hot = library.add(_speech(3.0) * 3.0, CLIP_SAMPLE_RATE, "hot")
    assert not hot.was_quiet
    assert float(np.max(np.abs(library.load(hot.id)[0]))) == pytest.approx(
        0.9, abs=0.01
    )
    assert ClipLibrary(tmp_path).get(quiet.id).source_peak_db == quiet.source_peak_db
    from_speakers = library.add(
        _speech(3.0) * 0.02, CLIP_SAMPLE_RATE, "quiet video", origin=ORIGIN_PLAYBACK
    )
    assert from_speakers.was_quiet and "video" in from_speakers.level_note
    assert "closer to the mic" not in from_speakers.level_note

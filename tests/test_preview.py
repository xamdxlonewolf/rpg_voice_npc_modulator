# Copyright (C) 2026 Michael Cobb
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from votr.dsp import DspEngine
from votr.preview import (
    keep_take,
    load_sample_take,
    play_on_speakers,
    render_take,
    save_last_take,
    speaker_devices,
)
from votr.session import Session
from votr.spikes.pitch_core import stretch_available
from votr.voice import Voice


def test_sample_take_is_mono_float() -> None:
    take = load_sample_take()
    assert take.ndim == 1
    assert take.size > 1000
    assert take.dtype == np.float32


def test_speaker_devices_skip_cable_input() -> None:
    devices = [
        {"name": "CABLE Input (VB-Audio Virtual Cable)", "max_output_channels": 2},
        {"name": "Speakers", "max_output_channels": 2},
    ]
    found = speaker_devices(devices)
    assert [dev["name"] for dev in found] == ["Speakers"]


@pytest.mark.skipif(not stretch_available(), reason="python-stretch required")
def test_render_equals_process_block_path() -> None:
    engine = DspEngine(prefer_rubband=False)
    take = load_sample_take()[: engine.block_size * 4]
    rendered = render_take(engine, take)
    engine2 = DspEngine(prefer_rubband=False)
    chunks = []
    padded = take
    rem = take.size % engine2.block_size
    if rem:
        padded = np.concatenate(
            [take, np.zeros(engine2.block_size - rem, dtype=np.float32)]
        )
    for i in range(0, padded.size, engine2.block_size):
        chunks.append(engine2.process_block(padded[i : i + engine2.block_size]))
    live = np.concatenate(chunks)[: take.size]
    np.testing.assert_allclose(rendered, live, atol=1e-5)


@pytest.mark.skipif(not stretch_available(), reason="python-stretch required")
def test_preview_other_voice_does_not_change_active(tmp_path: Path) -> None:
    session = Session(tmp_path)
    first = Voice.new("One")
    second = Voice.new("Two")
    second.params = {"pitch_semitones": -5.0}
    session.store.save(first)
    session.store.save(second)
    session.voices = session.store.load_all()
    session.set_active(first.id)
    session.take = load_sample_take()[:2048]
    from votr.app import create_application
    from votr.ui.preview_panel import PreviewPanel

    create_application(["votr-preview"])
    panel = PreviewPanel(session)
    panel.preview_other_voice(second.id)
    assert session.active_id == first.id


def test_keep_take_writes_named_wav(tmp_path: Path) -> None:
    take = np.linspace(-0.2, 0.2, 512, dtype=np.float32)
    save_last_take(tmp_path, take)
    path = keep_take(tmp_path, take, "grimjaw line")
    assert path.name == "grimjaw_line.wav"
    assert path.exists()
    assert (tmp_path / "takes" / "last.wav").exists()


def test_play_without_speakers_returns_false() -> None:
    assert play_on_speakers(np.zeros(64, dtype=np.float32)) is False

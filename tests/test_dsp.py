# Copyright (C) 2026 Michael Cobb
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import numpy as np
import pytest

from votr.dsp import DspEngine
from votr.macros import load_macros
from votr.session import Session
from votr.spikes.pitch_core import stretch_available

pytestmark = pytest.mark.skipif(
    not stretch_available(), reason="python-stretch required for DSP Engine tests"
)


def test_silence_in_silence_out() -> None:
    engine = DspEngine(prefer_rubband=False)
    silent = np.zeros(engine.block_size, dtype=np.float32)
    # Flush shifter memory, then a silent block must stay near silent.
    for _ in range(8):
        engine.process_block(silent)
    out = engine.process_block(silent)
    assert float(np.max(np.abs(out))) < 0.02


def test_rms_is_bounded() -> None:
    engine = DspEngine(prefer_rubband=False)
    engine.set_params({"growl": 1.0, "room": 0.5, "hollow": 0.4})
    t = np.arange(engine.block_size, dtype=np.float32) / engine.sample_rate
    loud = (0.95 * np.sin(2 * np.pi * 180 * t)).astype(np.float32)
    out = engine.process_block(loud)
    assert float(np.sqrt(np.mean(out * out))) < 1.0
    assert float(np.max(np.abs(out))) <= 1.0 + 1e-3


def test_latency_frames_reported() -> None:
    engine = DspEngine(prefer_rubband=False)
    assert engine.latency_frames() > 0


def test_render_uses_process_block() -> None:
    engine = DspEngine(prefer_rubband=False)
    take = np.linspace(-0.2, 0.2, engine.block_size * 3 + 40, dtype=np.float32)
    rendered = engine.render(take)
    assert rendered.shape == take.shape
    assert rendered.dtype == np.float32


def test_apply_macro_patches_only_defined_keys() -> None:
    engine = DspEngine(macros=load_macros(), prefer_rubband=False)
    engine.set_params({"room": 0.8, "growl": 0.0})
    engine.apply_macro("gravelly")
    params = engine.params()
    assert params["growl"] == pytest.approx(0.55)
    assert "room" not in load_macros()["gravelly"].params
    assert params["room"] == pytest.approx(0.8)


def _tone(hz: float, seconds: float, rate: int) -> np.ndarray:
    t = np.arange(int(rate * seconds), dtype=np.float32) / rate
    return (0.35 * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def _estimate_f0(samples: np.ndarray, rate: int) -> float:
    tail = samples[len(samples) // 3 :]
    tail = tail - float(np.mean(tail))
    corr = np.correlate(tail, tail, mode="full")
    corr = corr[corr.size // 2 :]
    min_lag = int(rate / 400)
    max_lag = int(rate / 80)
    peak = min_lag + int(np.argmax(corr[min_lag:max_lag]))
    return rate / float(peak)


def test_defaults_stay_close_to_dry_tone() -> None:
    engine = DspEngine(prefer_rubband=False)
    take = _tone(180.0, 0.4, engine.sample_rate)
    rendered = engine.render(take)
    corr = float(np.corrcoef(take, rendered)[0, 1])
    assert corr > 0.95
    assert float(np.max(np.abs(rendered))) < 0.7


def test_pitch_plus_minus_five_moves_f0() -> None:
    take = _tone(180.0, 0.5, 48_000)
    up = DspEngine(prefer_rubband=False)
    up.set_params({"pitch_semitones": 5.0})
    down = DspEngine(prefer_rubband=False)
    down.set_params({"pitch_semitones": -5.0})
    dry = DspEngine(prefer_rubband=False)
    f_dry = _estimate_f0(dry.render(take), 48_000)
    f_up = _estimate_f0(up.render(take), 48_000)
    f_down = _estimate_f0(down.render(take), 48_000)
    assert f_dry == pytest.approx(180.0, abs=12.0)
    assert f_up == pytest.approx(180.0 * 2 ** (5 / 12), abs=20.0)
    assert f_down == pytest.approx(180.0 * 2 ** (-5 / 12), abs=20.0)
    assert f_up > f_dry + 30
    assert f_down < f_dry - 25


def test_render_resets_streaming_leftovers() -> None:
    engine = DspEngine(prefer_rubband=False)
    engine.set_params({"growl": 1.0, "hollow": 0.8})
    noisy = _tone(180.0, 0.2, engine.sample_rate)
    engine.render(noisy)
    engine.set_params(
        {
            "growl": 0.0,
            "hollow": 0.0,
            "pitch_semitones": 0.0,
            "formant_semitones": 0.0,
        }
    )
    take = _tone(180.0, 0.3, engine.sample_rate)
    rendered = engine.render(take)
    assert float(np.corrcoef(take, rendered)[0, 1]) > 0.95


def test_new_voice_does_not_keep_previous_params(tmp_path) -> None:
    session = Session(tmp_path)
    session.engine.set_params({"growl": 0.9, "pitch_semitones": -7.0})
    session.edit_new()
    session.apply_draft_to_engine()
    params = session.engine.params()
    assert params["growl"] == pytest.approx(0.0)
    assert params["pitch_semitones"] == pytest.approx(0.0)

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


def _speech_like(seconds: float, rate: int, f0: float = 140.0) -> np.ndarray:
    n = int(rate * seconds)
    t = np.arange(n, dtype=np.float32) / rate
    voiced = np.zeros(n, dtype=np.float32)
    for harmonic, amp in enumerate(
        (0.24, 0.13, 0.08, 0.055, 0.04, 0.03, 0.022, 0.016), start=1
    ):
        voiced += amp * np.sin(2 * np.pi * f0 * harmonic * t)
    rng = np.random.default_rng(0)
    noise = rng.standard_normal(n).astype(np.float32)
    air = np.empty_like(noise)
    air[0] = noise[0]
    air[1:] = noise[1:] - 0.82 * noise[:-1]
    return (voiced + 0.045 * air).astype(np.float32)


def _rms(samples: np.ndarray) -> float:
    return float(np.sqrt(np.mean(samples * samples)) + 1e-12)


def _crest(samples: np.ndarray) -> float:
    return float(np.max(np.abs(samples)) / _rms(samples))


def _band_energy(samples: np.ndarray, rate: int, low_hz: float, high_hz: float) -> float:
    windowed = samples * np.hanning(samples.size)
    spec = np.abs(np.fft.rfft(windowed)) ** 2
    freqs = np.fft.rfftfreq(samples.size, 1.0 / rate)
    band = (freqs >= low_hz) & (freqs < high_hz)
    return float(np.mean(spec[band]) + 1e-18)


def _render_with(params: dict[str, float], take: np.ndarray) -> np.ndarray:
    engine = DspEngine(prefer_rubband=False)
    engine.set_params(params)
    return engine.render(take)


def test_tonality_bright_vs_dark_moves_spectrum() -> None:
    engine = DspEngine(prefer_rubband=False)
    take = _speech_like(0.45, engine.sample_rate)
    default = _render_with({"tonality": 0.5}, take)
    bright = _render_with({"tonality": 0.0}, take)
    dark = _render_with({"tonality": 1.0}, take)
    high_default = _band_energy(default, engine.sample_rate, 2000.0, 8000.0)
    high_bright = _band_energy(bright, engine.sample_rate, 2000.0, 8000.0)
    high_dark = _band_energy(dark, engine.sample_rate, 2000.0, 8000.0)
    assert high_bright > high_default * 1.35
    assert high_dark < high_default * 0.75


def test_growl_raises_saturation_crest() -> None:
    engine = DspEngine(prefer_rubband=False)
    take = _speech_like(0.4, engine.sample_rate)
    dry = _render_with({"growl": 0.0}, take)
    wet = _render_with({"growl": 1.0}, take)
    third_dry = _band_energy(dry, engine.sample_rate, 380.0, 460.0)
    third_wet = _band_energy(wet, engine.sample_rate, 380.0, 460.0)
    assert _crest(wet) < _crest(dry) * 0.92
    assert third_wet > third_dry * 1.25
    assert float(np.corrcoef(take, wet)[0, 1]) > 0.35


def test_hollow_comb_moves_spectrum() -> None:
    engine = DspEngine(prefer_rubband=False)
    take = _speech_like(0.4, engine.sample_rate)
    dry = _render_with({"hollow": 0.0}, take)
    wet = _render_with({"hollow": 1.0}, take)
    spec_dry = np.abs(np.fft.rfft(dry * np.hanning(dry.size)))
    spec_wet = np.abs(np.fft.rfft(wet * np.hanning(wet.size)))
    ripple_dry = float(np.std(np.log(spec_dry + 1e-8)))
    ripple_wet = float(np.std(np.log(spec_wet + 1e-8)))
    assert ripple_wet > ripple_dry * 1.08
    assert _rms(wet - dry) > 0.04
    assert float(np.corrcoef(take, wet)[0, 1]) > 0.25


def test_room_adds_audible_tail() -> None:
    engine = DspEngine(prefer_rubband=False)
    speech = _speech_like(0.28, engine.sample_rate)
    pad = np.zeros(int(engine.sample_rate * 0.22), dtype=np.float32)
    take = np.concatenate([speech, pad])
    dry = _render_with({"room": 0.0}, take)
    wet = _render_with({"room": 1.0}, take)
    tail = slice(speech.size, None)
    assert _rms(wet[tail]) > _rms(dry[tail]) * 3.0
    assert _rms(wet[tail]) > 0.008
    assert float(np.corrcoef(take[: speech.size], wet[: speech.size])[0, 1]) > 0.3


def test_distance_darker_and_quieter() -> None:
    engine = DspEngine(prefer_rubband=False)
    take = _speech_like(0.4, engine.sample_rate)
    dry = _render_with({"distance": 0.0}, take)
    wet = _render_with({"distance": 1.0}, take)
    high_dry = _band_energy(dry, engine.sample_rate, 1800.0, 7000.0)
    high_wet = _band_energy(wet, engine.sample_rate, 1800.0, 7000.0)
    assert _rms(wet) < _rms(dry) * 0.55
    assert high_wet < high_dry * 0.35


def test_breath_adds_noise_layer() -> None:
    engine = DspEngine(prefer_rubband=False)
    take = _speech_like(0.4, engine.sample_rate)
    dry = _render_with({"breath": 0.0}, take)
    wet = _render_with({"breath": 1.0}, take)
    high_dry = _band_energy(dry, engine.sample_rate, 3000.0, 10000.0)
    high_wet = _band_energy(wet, engine.sample_rate, 3000.0, 10000.0)
    assert _rms(wet) > _rms(dry) * 1.08
    assert high_wet > high_dry * 2.0
    assert float(np.corrcoef(take, wet)[0, 1]) > 0.45


def test_gate_silences_quiet_parts_keeps_speech() -> None:
    engine = DspEngine(prefer_rubband=False)
    rate = engine.sample_rate
    silence = np.zeros(int(rate * 0.2), dtype=np.float32)
    speech = _speech_like(0.3, rate)
    take = np.concatenate([silence, speech, silence])
    dry = _render_with({"gate": 0.0}, take)
    wet = _render_with({"gate": 1.0}, take)
    quiet = slice(0, silence.size)
    voiced = slice(silence.size + int(rate * 0.06), silence.size + speech.size)
    assert _rms(wet[quiet]) < _rms(dry[quiet]) * 0.35 + 0.002
    assert _rms(wet[quiet]) < 0.012
    assert _rms(wet[voiced]) > _rms(wet[quiet]) * 8.0
    assert _rms(wet[voiced]) > _rms(dry[voiced]) * 0.45


def test_body_thinner_vs_darker() -> None:
    engine = DspEngine(prefer_rubband=False)
    take = _speech_like(0.4, engine.sample_rate)
    dry = _render_with({"formant_semitones": 0.0}, take)
    thin = _render_with({"formant_semitones": 12.0}, take)
    dark = _render_with({"formant_semitones": -12.0}, take)

    def presence(audio: np.ndarray) -> float:
        high = _band_energy(audio, engine.sample_rate, 1800.0, 6000.0)
        low = _band_energy(audio, engine.sample_rate, 80.0, 400.0)
        return high / low

    assert presence(thin) > presence(dry) * 1.25
    assert presence(dark) < presence(dry) * 0.80

# Copyright (C) 2026 Michael Cobb
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import numpy as np
import pytest

from votr.dsp import DspEngine, active_rms, breath_air
from votr.engine import BaseEngine
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


# --- Slider character tests -------------------------------------------------
# Michael's Windows listen after #13: most sliders read as volume or static.
# These assert spectral shape, crest, harmonic content and noise floor with the
# level held near dry — RMS-only checks let "just louder" slip through.

RATE = 48_000
_FORMANTS = ((550.0, 80.0), (1400.0, 110.0), (2500.0, 160.0), (3400.0, 220.0))


def _vowel(
    seconds: float,
    *,
    f0: float = 120.0,
    peak: float = 0.3,
    syllables: bool = True,
    seed: int = 0,
) -> np.ndarray:
    """Harmonic vowel with real formant peaks and a 3.5 Hz syllable envelope."""
    n = int(RATE * seconds)
    t = np.arange(n, dtype=np.float64) / RATE
    rng = np.random.default_rng(seed)
    vibrato = 0.003 * np.sin(2 * np.pi * 5.0 * t) / (2 * np.pi * 5.0) * f0
    voiced = np.zeros(n)
    for k in range(1, int(7000 / f0)):
        freq = k * f0
        envelope = sum(
            1.0 / (1.0 + ((freq - centre) / width) ** 2) for centre, width in _FORMANTS
        )
        tilt = (f0 / freq) ** 1.0
        voiced += envelope * tilt * np.sin(2 * np.pi * k * (f0 * t + vibrato))
    air = rng.standard_normal(n)
    air = np.convolve(air, np.hanning(9) / np.sum(np.hanning(9)), mode="same")
    air = air - np.convolve(air, np.ones(48) / 48, mode="same")
    signal = voiced + 0.12 * air * np.max(np.abs(voiced))
    if syllables:
        signal *= (0.5 + 0.5 * np.sin(2 * np.pi * 3.5 * t - np.pi / 2)) ** 0.7
    return (signal / np.max(np.abs(signal)) * peak).astype(np.float32)


def _rms(samples: np.ndarray) -> float:
    return float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)) + 1e-12)


def _db(value: float) -> float:
    return float(20.0 * np.log10(max(value, 1e-12)))


def _crest(samples: np.ndarray) -> float:
    return float(np.max(np.abs(samples)) / _rms(samples))


def _spectrum(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    windowed = samples.astype(np.float64) * np.hanning(samples.size)
    return np.fft.rfftfreq(samples.size, 1.0 / RATE), np.abs(np.fft.rfft(windowed))


def _band_energy(samples: np.ndarray, low_hz: float, high_hz: float) -> float:
    freqs, spec = _spectrum(samples)
    band = (freqs >= low_hz) & (freqs < high_hz)
    return float(np.mean(spec[band] ** 2) + 1e-18)


def _band_db(wet: np.ndarray, dry: np.ndarray, low_hz: float, high_hz: float) -> float:
    return float(
        10.0
        * np.log10(
            _band_energy(wet, low_hz, high_hz) / _band_energy(dry, low_hz, high_hz)
        )
    )


def _log_envelope(samples: np.ndarray) -> tuple[np.ndarray, float]:
    """Cepstrally smoothed log spectrum on a uniform log-frequency grid."""
    freqs, spec = _spectrum(samples)
    cep = np.fft.irfft(np.log(spec + 1e-9))
    lifter = int(RATE / 350.0)
    cep[lifter:-lifter] = 0.0
    envelope = np.fft.rfft(cep).real
    grid = np.linspace(np.log2(150.0), np.log2(8000.0), 480)
    step_semitones = float((grid[1] - grid[0]) * 12.0)
    return np.interp(grid, np.log2(freqs[1:]), envelope[1:]), step_semitones


def _envelope_shift_semitones(wet: np.ndarray, dry: np.ndarray) -> float:
    """How far the formant envelope moved, in semitones (best log-f alignment)."""
    env_wet, step = _log_envelope(wet)
    env_dry, _ = _log_envelope(dry)
    best_shift = 0.0
    best_score = -2.0
    for k in range(-120, 121):
        if k >= 0:
            a, b = env_wet[k:], env_dry[: env_wet.size - k]
        else:
            a, b = env_wet[:k], env_dry[-k:]
        a = a - np.mean(a)
        b = b - np.mean(b)
        score = float(np.dot(a, b) / np.sqrt(np.dot(a, a) * np.dot(b, b) + 1e-12))
        if score > best_score:
            best_score = score
            best_shift = k * step
    return best_shift


def _level_delta_db(wet: np.ndarray, dry: np.ndarray) -> float:
    return _db(active_rms(wet, RATE)) - _db(active_rms(dry, RATE))


def _render_with(params: dict[str, float], take: np.ndarray) -> np.ndarray:
    engine = DspEngine(prefer_rubband=False)
    engine.set_params(params)
    return engine.render(take)


def _with_gaps(
    speech: np.ndarray, gap_seconds: float, floor: float = 0.0
) -> np.ndarray:
    gap = np.zeros(int(RATE * gap_seconds), dtype=np.float32)
    take = np.concatenate([gap, speech, gap])
    if floor > 0.0:
        hiss = np.random.default_rng(7).standard_normal(take.size) * floor
        take = (take + hiss).astype(np.float32)
    return take


def test_active_rms_ignores_silence_and_tails() -> None:
    speech = _vowel(0.4, syllables=False)
    padded = _with_gaps(speech, 0.8)
    assert active_rms(padded, RATE) == pytest.approx(active_rms(speech, RATE), rel=0.15)
    assert active_rms(np.zeros(4800, dtype=np.float32), RATE) == 0.0


def test_zero_is_off_for_unipolar_sliders() -> None:
    take = _vowel(0.3)
    reference = _render_with({}, take)
    for key in ("growl", "hollow", "room", "distance", "breath", "gate"):
        np.testing.assert_allclose(_render_with({key: 0.0}, take), reference, atol=1e-6)


def test_body_moves_formants_not_pitch_or_level() -> None:
    take = _vowel(0.8)
    dry = _render_with({}, take)
    chesty = _render_with({"formant_semitones": -8.0}, take)
    thin = _render_with({"formant_semitones": 8.0}, take)
    assert _envelope_shift_semitones(chesty, dry) == pytest.approx(-8.0, abs=1.5)
    assert _envelope_shift_semitones(thin, dry) == pytest.approx(8.0, abs=1.5)
    assert _envelope_shift_semitones(dry, dry) == pytest.approx(0.0, abs=0.2)
    for wet in (chesty, thin):
        assert _estimate_f0(wet, RATE) == pytest.approx(
            _estimate_f0(dry, RATE), rel=0.05
        )
        assert abs(_level_delta_db(wet, dry)) < 1.5


def test_body_negative_is_the_mirror_of_positive() -> None:
    take = _vowel(0.6)
    dry = _render_with({}, take)
    chesty = _render_with({"formant_semitones": -8.0}, take)
    thin = _render_with({"formant_semitones": 8.0}, take)
    assert _band_db(chesty, dry, 2000.0, 6000.0) < -4.0
    assert _band_db(thin, dry, 2000.0, 6000.0) > 1.5
    assert _band_db(chesty, dry, 100.0, 450.0) > _band_db(thin, dry, 100.0, 450.0) + 2.0


def test_tonality_is_a_tilt_not_a_volume_knob() -> None:
    take = _vowel(0.6)
    dry = _render_with({"tonality": 0.5}, take)
    np.testing.assert_allclose(dry, _render_with({}, take), atol=1e-6)
    bright = _render_with({"tonality": 0.0}, take)
    dark = _render_with({"tonality": 1.0}, take)
    assert _band_db(bright, dry, 2000.0, 8000.0) > 4.0
    assert _band_db(bright, dry, 80.0, 400.0) < -3.0
    assert _band_db(dark, dry, 2000.0, 8000.0) < -7.0
    assert _band_db(dark, dry, 80.0, 400.0) > 1.0
    assert abs(_level_delta_db(bright, dry)) < 1.0
    assert abs(_level_delta_db(dark, dry)) < 1.0
    half_bright = _render_with({"tonality": 0.25}, take)
    assert (
        1.0
        < _band_db(half_bright, dry, 2000.0, 8000.0)
        < _band_db(bright, dry, 2000.0, 8000.0)
    )


def test_growl_adds_harmonic_grit_at_matched_level() -> None:
    t = np.arange(int(RATE * 0.5), dtype=np.float64) / RATE
    tone = (0.2 * np.sin(2 * np.pi * 150.0 * t)).astype(np.float32)
    dry = _render_with({}, tone)
    wet = _render_with({"growl": 1.0}, tone)
    freqs, spec_dry = _spectrum(dry)
    _, spec_wet = _spectrum(wet)

    def harmonic_db(spec: np.ndarray, k: int) -> float:
        fundamental = float(np.max(spec[(freqs > 130.0) & (freqs < 170.0)]))
        band = (freqs > 150.0 * k - 20.0) & (freqs < 150.0 * k + 20.0)
        return float(20.0 * np.log10(np.max(spec[band]) / fundamental + 1e-12))

    assert harmonic_db(spec_dry, 2) < -60.0 and harmonic_db(spec_dry, 3) < -60.0
    assert harmonic_db(spec_wet, 2) > -30.0
    assert harmonic_db(spec_wet, 3) > -25.0
    assert abs(_level_delta_db(wet, dry)) < 1.0

    speech = _vowel(0.6)
    dry_speech = _render_with({}, speech)
    wet_speech = _render_with({"growl": 1.0}, speech)
    assert _crest(wet_speech) < _crest(dry_speech) * 0.8
    assert abs(_level_delta_db(wet_speech, dry_speech)) < 1.0
    assert float(np.corrcoef(dry_speech, wet_speech)[0, 1]) > 0.6


def test_hollow_is_a_cupped_cavity_not_a_metallic_ring() -> None:
    take = _vowel(0.6)
    dry = _render_with({}, take)
    wet = _render_with({"hollow": 1.0}, take)
    mild = _render_with({"hollow": 0.5}, take)
    assert _band_db(wet, dry, 80.0, 400.0) < -5.0
    assert _band_db(wet, dry, 700.0, 1500.0) > 2.0
    assert _band_db(wet, dry, 3000.0, 8000.0) < -1.5
    assert abs(_level_delta_db(wet, dry)) < 1.0
    assert abs(_level_delta_db(mild, dry)) < 1.0
    assert _band_db(wet, dry, 80.0, 400.0) < _band_db(mild, dry, 80.0, 400.0) < -1.0
    assert _estimate_f0(mild, RATE) == pytest.approx(_estimate_f0(dry, RATE), rel=0.05)


def test_room_adds_audible_tail() -> None:
    speech = _vowel(0.3, syllables=False)
    pad = np.zeros(int(RATE * 0.25), dtype=np.float32)
    take = np.concatenate([speech, pad])
    dry = _render_with({"room": 0.0}, take)
    wet = _render_with({"room": 1.0}, take)
    tail = slice(speech.size, None)
    assert _rms(wet[tail]) > _rms(dry[tail]) * 3.0
    assert _rms(wet[tail]) > 0.008
    assert float(np.corrcoef(take[: speech.size], wet[: speech.size])[0, 1]) > 0.3


def test_distance_is_gently_quieter_and_darker() -> None:
    take = _vowel(0.6)
    dry = _render_with({}, take)
    near = _render_with({"distance": 0.25}, take)
    mid = _render_with({"distance": 0.5}, take)
    far = _render_with({"distance": 1.0}, take)
    assert -2.5 < _level_delta_db(near, dry) < -0.3
    assert -5.5 < _level_delta_db(mid, dry) < -2.0
    assert -13.0 < _level_delta_db(far, dry) < -7.0

    def balance(audio: np.ndarray) -> float:
        return _band_energy(audio, 2000.0, 8000.0) / _band_energy(audio, 100.0, 500.0)

    assert balance(near) < balance(dry) * 0.9
    assert balance(mid) < balance(near)
    assert balance(far) < balance(mid) * 0.3


def _frame_levels(samples: np.ndarray, frame: int) -> np.ndarray:
    count = samples.size // frame
    frames = samples[: count * frame].reshape(count, frame).astype(np.float64)
    return np.sqrt(np.mean(frames * frames, axis=1))


def test_breath_is_voice_shaped_air_not_white_noise() -> None:
    speech = _vowel(0.9)
    air = breath_air(speech, RATE)
    white = np.random.default_rng(0).standard_normal(speech.size).astype(np.float32)
    env_voice, _ = _log_envelope(speech)
    env_air, _ = _log_envelope(air)
    env_white, _ = _log_envelope(white)
    # The air carries the voice's own formant shape; static does not.
    assert float(np.corrcoef(env_air, env_voice)[0, 1]) > 0.7
    assert float(np.corrcoef(env_white, env_voice)[0, 1]) < 0.3
    # ...and it rides the syllables sample-tight, not a slow bed under them.
    frame = int(RATE * 0.02)
    air_level = _frame_levels(air, frame)
    voice_level = _frame_levels(speech, frame)
    assert float(np.corrcoef(air_level, voice_level)[0, 1]) > 0.9
    order = np.argsort(voice_level)
    quarter = order.size // 4
    assert np.mean(air_level[order[-quarter:]]) > 5.0 * np.mean(
        air_level[order[:quarter]]
    )


def test_breath_never_raises_the_noise_floor() -> None:
    speech = _vowel(0.9)
    gap = slice(0, int(RATE * 0.35))
    voiced = slice(int(RATE * 0.4), int(RATE * 0.4) + speech.size)
    for floor in (0.0, 0.001):
        take = _with_gaps(speech, 0.4, floor=floor)
        dry = _render_with({}, take)
        for amount in (0.3, 1.0):
            wet = _render_with({"breath": amount}, take)
            assert _rms(wet[gap]) <= _rms(dry[gap]) * 1.05 + 1e-7
            assert abs(_level_delta_db(wet, dry)) < 1.0
        wet = _render_with({"breath": 1.0}, take)
        assert float(np.corrcoef(wet[voiced], dry[voiced])[0, 1]) < 0.9
        assert float(np.corrcoef(wet[voiced], dry[voiced])[0, 1]) > 0.5
    # Digital silence in stays digital silence out.
    silent = np.zeros(int(RATE * 0.5), dtype=np.float32)
    assert float(np.max(np.abs(_render_with({"breath": 1.0}, silent)))) < 1e-6


def test_breath_streaming_path_matches_render_floor_rule() -> None:
    speech = _vowel(0.9)
    take = _with_gaps(speech, 0.4, floor=0.001)
    engine = DspEngine(prefer_rubband=False)
    engine.set_params({"breath": 1.0})
    engine.reset()
    live = BaseEngine.render(engine, take)
    gap = slice(0, int(RATE * 0.35))
    voiced = slice(int(RATE * 0.4), int(RATE * 0.4) + speech.size)
    assert _rms(live[gap]) <= _rms(take[gap]) * 1.05
    assert float(np.corrcoef(live[voiced], take[voiced])[0, 1]) > 0.5
    assert _rms(live[voiced]) > _rms(take[voiced]) * 0.6


def test_gate_mutes_quiet_parts_and_leaves_speech_untouched() -> None:
    speech = _vowel(0.9, peak=0.15)
    take = _with_gaps(speech, 0.6, floor=0.0015)
    dry = _render_with({}, take)
    gap = slice(0, int(RATE * 0.55))
    tail = slice(int(RATE * 0.6) + speech.size + int(RATE * 0.35), None)
    voiced = slice(
        int(RATE * 0.6) + int(RATE * 0.05),
        int(RATE * 0.6) + speech.size - int(RATE * 0.05),
    )
    assert _db(_rms(dry[gap])) > -60.0
    for amount in (0.35, 0.7):
        wet = _render_with({"gate": amount}, take)
        assert _db(_rms(wet[gap])) < -75.0
        assert _db(_rms(wet[tail])) < _db(_rms(dry[tail])) - 12.0
        assert float(np.corrcoef(dry[voiced], wet[voiced])[0, 1]) > 0.995
        assert abs(_db(_rms(wet[voiced])) - _db(_rms(dry[voiced]))) < 0.5


def test_gate_at_whisper_setting_keeps_a_quiet_voice() -> None:
    speech = _vowel(0.9, peak=0.08)
    take = _with_gaps(speech, 0.5, floor=0.0005)
    dry = _render_with({}, take)
    wet = _render_with({"gate": 0.35}, take)
    voiced = slice(
        int(RATE * 0.5) + int(RATE * 0.05),
        int(RATE * 0.5) + speech.size - int(RATE * 0.05),
    )
    assert float(np.corrcoef(dry[voiced], wet[voiced])[0, 1]) > 0.99
    assert _db(_rms(wet[: int(RATE * 0.45)])) < -75.0

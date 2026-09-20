# Copyright (C) 2026 Michael Cobb
# SPDX-License-Identifier: GPL-3.0-or-later

"""DSP Engine v1: gate → pitch/formant → texture → room → limiter."""

from __future__ import annotations

from typing import Any

import numpy as np

from votr.engine import (
    DEFAULT_SAMPLE_RATE,
    BaseEngine,
    EngineCapabilities,
    ParameterSpec,
)
from votr.macros import Macro
from votr.spikes.pitch_core import rubband_available, stretch_available
from votr.voice import DSP_ENGINE_ID

SCHEMA: tuple[ParameterSpec, ...] = (
    ParameterSpec("pitch_semitones", "Pitch (semitones)", -12.0, 12.0, 0.0),
    ParameterSpec("formant_semitones", "Body (− chesty · + thin)", -12.0, 12.0, 0.0),
    ParameterSpec("tonality", "Tonality (bright ← → dark)", 0.0, 1.0, 0.5),
    ParameterSpec("growl", "Growl (0 = off)", 0.0, 1.0, 0.0),
    ParameterSpec("hollow", "Hollow (0 = off)", 0.0, 1.0, 0.0),
    ParameterSpec("room", "Room (0 = off)", 0.0, 1.0, 0.0),
    ParameterSpec("distance", "Distance (0 = close)", 0.0, 1.0, 0.0),
    ParameterSpec("breath", "Breath (0 = off)", 0.0, 1.0, 0.0),
    ParameterSpec("gate", "Gate (0 = off)", 0.0, 1.0, 0.0),
)

_DEFAULTS = {spec.key: spec.default for spec in SCHEMA}
# python-stretch only transposes when the process hop is ≥ outputLatency (~30 ms).
STRETCH_HOP = 1440
# Envelope followers run on short sub-frames and interpolate gain per sample,
# so gates and breath never zipper or modulate inside a waveform cycle.
_ENV_FRAME = 64
_LEVEL_FRAME_SECONDS = 0.020
_MAX_MATCH_GAIN = 4.0
# Body on python-stretch: STFT spectral-envelope warp (no formant API in binding).
_FORMANT_FFT = 2048
_FORMANT_SMOOTH_HZ = 350.0
_FORMANT_MAX_DB = 24.0
# Breath: whisperised copy of the voice (random-phase STFT), never free noise.
_BREATH_FFT = 1024
_BREATH_FLOOR_DB = -50.0
_BREATH_SEED = 1234


def _plugin_mono(array: np.ndarray) -> np.ndarray:
    samples = np.asarray(array, dtype=np.float32)
    if samples.ndim == 2:
        samples = samples[0] if samples.shape[0] <= samples.shape[1] else samples[:, 0]
    return samples


def _semitone_ratio(semitones: float) -> float:
    return float(2.0 ** (semitones / 12.0))


def _db_to_gain(db: float) -> float:
    return float(10.0 ** (db / 20.0))


def _frame_rms(samples: np.ndarray, frame: int) -> np.ndarray:
    """RMS per ``frame``-sample sub-frame; a ragged tail keeps its own frame."""
    n = samples.size
    count = max(1, -(-n // frame))
    padded = np.zeros(count * frame, dtype=np.float64)
    padded[:n] = samples
    blocks = padded.reshape(count, frame)
    return np.sqrt(np.mean(blocks * blocks, axis=1))


def _per_sample(values: np.ndarray, previous: float, n: int, frame: int) -> np.ndarray:
    """Linearly ramp sub-frame values across their samples (no gain steps)."""
    knots = np.concatenate([[previous], values])
    ends = np.arange(knots.size) * frame
    return np.interp(np.arange(n), ends, knots).astype(np.float32)


def active_rms(samples: np.ndarray, sample_rate: int) -> float:
    """RMS of the louder frames only, so silence, gates and tails do not skew it."""
    frame = max(1, int(sample_rate * _LEVEL_FRAME_SECONDS))
    levels = _frame_rms(np.asarray(samples, dtype=np.float64), frame)
    loudest = float(np.max(levels)) if levels.size else 0.0
    if loudest <= 1e-6:
        return 0.0
    kept = levels[levels >= loudest * 0.1]
    return float(np.sqrt(np.mean(kept * kept)))


def _stft(x: np.ndarray, n_fft: int, hop: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Windowed frames of ``x`` (padded by one FFT each side). Returns frames,
    window and padded length so ``_istft`` can undo it exactly."""
    window = np.hanning(n_fft + 1)[:-1]
    padded = np.pad(x, (n_fft, n_fft + hop))
    count = 1 + (padded.size - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(count)[:, None]
    return padded[idx] * window, window, padded.size


def _istft(
    frames: np.ndarray, window: np.ndarray, hop: int, padded_size: int, size: int
) -> np.ndarray:
    n_fft = window.size
    out = np.zeros(padded_size, dtype=np.float64)
    norm = np.zeros(padded_size, dtype=np.float64)
    for i in range(frames.shape[0]):
        start = i * hop
        out[start : start + n_fft] += frames[i] * window
        norm[start : start + n_fft] += window * window
    out = out / np.maximum(norm, 1e-3)
    return out[n_fft : n_fft + size]


def formant_shift(
    samples: np.ndarray, semitones: float, sample_rate: int
) -> np.ndarray:
    """Move the spectral envelope (vocal-tract size) without touching pitch.

    STFT frames are split into a cepstrally smoothed envelope and fine harmonic
    structure; the envelope is stretched along frequency by the semitone ratio
    and re-imposed. Positive = smaller/thinner tract, negative = larger/chestier.
    """
    ratio = _semitone_ratio(semitones)
    x = np.asarray(samples, dtype=np.float64)
    if x.size == 0 or abs(ratio - 1.0) < 1e-4:
        return np.asarray(samples, dtype=np.float32)
    n_fft = _FORMANT_FFT
    hop = n_fft // 4
    frames, window, padded_size = _stft(x, n_fft, hop)
    spec = np.fft.rfft(frames, axis=1)
    mag = np.abs(spec)
    log_mag = np.log(mag + 1e-9)
    cep = np.fft.irfft(log_mag, n=n_fft, axis=1)
    lifter = max(8, int(sample_rate / _FORMANT_SMOOTH_HZ))
    cep[:, lifter:-lifter] = 0.0
    envelope = np.fft.rfft(cep, n=n_fft, axis=1).real
    bins = np.arange(envelope.shape[1], dtype=np.float64)
    source = np.clip(bins / ratio, 0.0, bins[-1])
    warped = np.empty_like(envelope)
    for i in range(envelope.shape[0]):
        warped[i] = np.interp(source, bins, envelope[i])
    limit = _FORMANT_MAX_DB / 20.0 * np.log(10.0)
    correction = np.clip(warped - envelope, -limit, limit)
    spec = spec * np.exp(correction)
    frames = np.fft.irfft(spec, n=n_fft, axis=1)
    return _istft(frames, window, hop, padded_size, x.size).astype(np.float32)


def _air_weight(n_fft: int, sample_rate: int) -> np.ndarray:
    """Aspiration sits above the fundamental and rolls off at the top."""
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)
    rise = np.clip((freqs - 300.0) / (2500.0 - 300.0), 0.0, 1.0) ** 0.5
    return rise / (1.0 + (freqs / 8000.0) ** 4)


def _whisper_frames(
    frames: np.ndarray, weight: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Whispered copy of windowed frames: same magnitude spectrum, random phase.

    Each frame keeps the voice's own formant shape and level, so the result is
    aspiration that follows the words. Frames at or below a quiet-room floor
    are zeroed so silence never gains a hiss bed.
    """
    voice_rms = np.sqrt(np.mean(frames * frames, axis=1))
    spec = np.fft.rfft(frames, axis=1)
    phase = rng.uniform(0.0, 2.0 * np.pi, spec.shape)
    air_spec = np.abs(spec) * weight[None, :] * np.exp(1j * phase)
    air = np.fft.irfft(air_spec, n=frames.shape[1], axis=1)
    air_rms = np.sqrt(np.mean(air * air, axis=1))
    level_db = 20.0 * np.log10(voice_rms + 1e-9)
    presence = np.clip((level_db - _BREATH_FLOOR_DB) / 12.0, 0.0, 1.0)
    scale = np.where(air_rms > 1e-9, voice_rms / np.maximum(air_rms, 1e-9), 0.0)
    return air * (np.minimum(scale, 8.0) * presence)[:, None]


def breath_air(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    """Whole-Take aspiration layer derived from the voice itself (see above)."""
    x = np.asarray(samples, dtype=np.float64)
    if x.size == 0:
        return np.asarray(samples, dtype=np.float32)
    n_fft = _BREATH_FFT
    hop = n_fft // 4
    frames, window, padded_size = _stft(x, n_fft, hop)
    rng = np.random.default_rng(_BREATH_SEED)
    air = _whisper_frames(frames, _air_weight(n_fft, sample_rate), rng)
    return _istft(air, window, hop, padded_size, x.size).astype(np.float32)


def mix_breath(voice: np.ndarray, air: np.ndarray, amount: float) -> np.ndarray:
    """Voice steps back a little as the air comes up; air ≤ 0.9× the voice."""
    return (voice * (1.0 - 0.35 * amount) + air * (0.9 * amount)).astype(np.float32)


class DspEngine(BaseEngine):
    """CPU Engine that modulates the GM's own voice (ADR-0004)."""

    engine_id = DSP_ENGINE_ID

    def __init__(
        self,
        *,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        macros: dict[str, Macro] | None = None,
        prefer_rubband: bool = True,
        block_size: int | None = None,
    ) -> None:
        self._use_rubband = bool(prefer_rubband and rubband_available())
        if not self._use_rubband and not stretch_available():
            raise RuntimeError("DSP Engine needs python-stretch or rubband")
        default_block = 512 if self._use_rubband else 256
        chosen = int(block_size) if block_size else default_block
        super().__init__(block_size=chosen, sample_rate=sample_rate)
        self._params = dict(_DEFAULTS)
        self._macro_defs = macros or {}
        self._room_delay = np.zeros(max(8, int(sample_rate * 0.090)), dtype=np.float32)
        self._room_i = 0
        self._shifter: Any = None
        hop = max(chosen, STRETCH_HOP)
        self._stretch_in = np.zeros((1, hop), dtype=np.float32)
        self._stretch_pending = np.zeros(0, dtype=np.float32)
        self._stretch_ready = np.zeros(0, dtype=np.float32)
        self._live_out = np.zeros(chosen, dtype=np.float32)
        self._rng = np.random.default_rng()
        self._fade_left = 0
        self._fade_total = 0
        self._fade_prev: np.ndarray | None = None
        self._defer_distance_gain = False
        self._reset_followers()
        self._init_effects()
        self._update_filters()
        self._rebuild_shifter()

    def _reset_followers(self) -> None:
        self._gate_env = 0.0
        # Start closed: a Take that opens on silence stays silent from sample 0.
        self._gate_gain = 0.0
        self._gate_open = False
        self._gate_hold = 0
        self._breath_prev = np.zeros(self.block_size, dtype=np.float32)
        self._breath_tail = np.zeros(self.block_size, dtype=np.float64)
        self._growl_phase = 0.0

    def _init_effects(self) -> None:
        import pedalboard as pb

        self._reverb = pb.Reverb(
            room_size=0.45,
            damping=0.32,
            wet_level=0.0,
            dry_level=1.0,
            width=0.55,
        )
        self._limiter = pb.Limiter(threshold_db=-1.5, release_ms=50)
        # Body fallback for the streaming path (render uses a true formant shift).
        self._body_low = pb.LowShelfFilter(cutoff_frequency_hz=380.0, gain_db=0.0)
        self._body_high = pb.HighShelfFilter(cutoff_frequency_hz=2600.0, gain_db=0.0)
        # Tonality: a tilt pivoting near 1 kHz — lows and highs move opposite ways.
        self._tone_low = pb.LowShelfFilter(cutoff_frequency_hz=650.0, gain_db=0.0)
        self._tone_high = pb.HighShelfFilter(cutoff_frequency_hz=2000.0, gain_db=0.0)
        self._growl_pre = pb.HighpassFilter(cutoff_frequency_hz=170.0)
        self._growl_post = pb.LowpassFilter(cutoff_frequency_hz=6500.0)
        self._hollow_hp = pb.HighpassFilter(cutoff_frequency_hz=380.0)
        self._hollow_hp2 = pb.HighpassFilter(cutoff_frequency_hz=380.0)
        self._hollow_lp = pb.LowpassFilter(cutoff_frequency_hz=2600.0)
        self._hollow_lp2 = pb.LowpassFilter(cutoff_frequency_hz=2600.0)
        self._hollow_peak = pb.PeakFilter(
            cutoff_frequency_hz=1050.0, gain_db=8.0, q=1.4
        )
        self._hollow_comb = pb.Delay(delay_seconds=0.009, feedback=0.3, mix=0.3)
        self._breath_window = np.sqrt(np.hanning(2 * self.block_size + 1)[:-1])
        self._breath_weight = _air_weight(2 * self.block_size, self.sample_rate)
        self._dist_lp1 = pb.LowpassFilter(cutoff_frequency_hz=12000.0)
        self._dist_lp2 = pb.LowpassFilter(cutoff_frequency_hz=12000.0)

    def _plugins(self) -> tuple[Any, ...]:
        return (
            self._reverb,
            self._limiter,
            self._body_low,
            self._body_high,
            self._tone_low,
            self._tone_high,
            self._growl_pre,
            self._growl_post,
            self._hollow_hp,
            self._hollow_hp2,
            self._hollow_lp,
            self._hollow_lp2,
            self._hollow_peak,
            self._hollow_comb,
            self._dist_lp1,
            self._dist_lp2,
        )

    def _run(self, plugin: Any, work: np.ndarray) -> np.ndarray:
        return _plugin_mono(
            plugin.process(
                work, self.sample_rate, buffer_size=self.block_size, reset=False
            )
        )

    def reset(self) -> None:
        """Clear streaming leftovers before rendering a Take."""
        self._room_delay.fill(0)
        self._room_i = 0
        self._fade_left = 0
        self._fade_prev = None
        self._reset_followers()
        self._stretch_pending = np.zeros(0, dtype=np.float32)
        self._stretch_ready = np.zeros(0, dtype=np.float32)
        self._rebuild_shifter()
        silent = np.zeros(self.block_size, dtype=np.float32)
        for plugin in self._plugins():
            plugin.process(
                silent, self.sample_rate, buffer_size=self.block_size, reset=True
            )

    def _stretch_offline(
        self, samples: np.ndarray, *, pitch: float, formant: float
    ) -> np.ndarray:
        """Whole-Take pitch (python-stretch) then Body (spectral-envelope warp)."""
        work = samples
        if abs(pitch) >= 0.05:
            self._rebuild_shifter()
            framed = np.ascontiguousarray(work.reshape(1, -1))
            shifted = self._shifter.process(framed)
            work = np.asarray(shifted, dtype=np.float32).reshape(-1)
            if work.size >= samples.size:
                work = work[: samples.size]
            else:
                work = np.pad(work, (0, samples.size - work.size))
        if abs(formant) >= 0.05:
            work = formant_shift(work, formant, self.sample_rate)
        return work

    def render(self, take: np.ndarray) -> np.ndarray:
        """Render a Take from a clean state — not leftover stream memory."""
        self.reset()
        samples = np.asarray(take, dtype=np.float32).reshape(-1)
        if samples.size == 0:
            return samples.copy()
        dry_level = active_rms(samples, self.sample_rate)
        pitch = float(self._params["pitch_semitones"])
        formant = float(self._params["formant_semitones"])
        saved = dict(self._params)
        self._defer_distance_gain = True
        try:
            if not self._use_rubband and (abs(pitch) >= 0.05 or abs(formant) >= 0.05):
                samples = self._stretch_offline(samples, pitch=pitch, formant=formant)
                self._params["pitch_semitones"] = 0.0
                self._params["formant_semitones"] = 0.0
                self._update_filters()
            breath = float(self._params["breath"])
            if breath >= 0.01:
                air = breath_air(samples, self.sample_rate)
                samples = mix_breath(samples, air, breath)
                self._params["breath"] = 0.0
            out = super().render(samples)
        finally:
            self._params.update(saved)
            self._defer_distance_gain = False
            self._update_filters()
        out = self._match_level(out, dry_level)
        out = (out * self._distance_gain()).astype(np.float32)
        room = float(self._params["room"])
        if room > 0.01:
            out = self._apply_room_offline(out, room)
        peak = float(np.max(np.abs(out))) if out.size else 0.0
        if peak > 0.98:
            out = _plugin_mono(self._limiter.process(out, self.sample_rate, reset=True))
        return np.clip(out, -1.0, 1.0).astype(np.float32)

    def _match_level(self, out: np.ndarray, dry_level: float) -> np.ndarray:
        """Bring the effect chain back to the Take's level (Distance is separate)."""
        wet_level = active_rms(out, self.sample_rate)
        if dry_level <= 1e-6 or wet_level <= 1e-6:
            return out
        gain = min(_MAX_MATCH_GAIN, max(1.0 / _MAX_MATCH_GAIN, dry_level / wet_level))
        return (out * gain).astype(np.float32)

    def _rebuild_shifter(self) -> None:
        pitch = float(self._params["pitch_semitones"])
        formant = float(self._params["formant_semitones"])
        if self._use_rubband:
            import rubband

            options = rubband.LiveOptions(formant=rubband.LiveFormantOption.preserved)
            self._shifter = rubband.LiveShifter(self.sample_rate, 1, options=options)
            self._shifter.set_pitch_scale(_semitone_ratio(pitch))
            self._shifter.set_formant_scale(_semitone_ratio(formant))
            return
        import python_stretch as ps

        block = max(1, int(self.sample_rate * 0.060))
        self._shifter = ps.Signalsmith.Stretch()
        self._shifter.configure(1, block, max(1, block // 4))
        self._shifter.setTimeFactor(1.0)
        # Clean transpose only — no tonality-limit formant (binding has none).
        self._shifter.setTransposeSemitones(pitch)
        if self._stretch_in.shape[1] != STRETCH_HOP:
            self._stretch_in = np.zeros((1, STRETCH_HOP), dtype=np.float32)

    def _stretch_block(self, work: np.ndarray) -> np.ndarray:
        """Pitch-shift one I/O block. Stretch needs a ≥30 ms hop to transpose."""
        if work.size >= STRETCH_HOP:
            self._stretch_in[0, : work.size] = work
            return self._shifter.process(self._stretch_in[:, : work.size])[0].copy()
        self._stretch_pending = np.concatenate([self._stretch_pending, work])
        while self._stretch_pending.size >= STRETCH_HOP:
            frame = self._stretch_pending[:STRETCH_HOP]
            self._stretch_pending = self._stretch_pending[STRETCH_HOP:]
            self._stretch_in[0] = frame
            shifted = self._shifter.process(self._stretch_in)[0]
            self._stretch_ready = np.concatenate([self._stretch_ready, shifted])
        if self._stretch_ready.size >= work.size:
            out = self._stretch_ready[: work.size]
            self._stretch_ready = self._stretch_ready[work.size :]
            return out.astype(np.float32)
        pad = work.size - self._stretch_ready.size
        out = np.concatenate([np.zeros(pad, dtype=np.float32), self._stretch_ready])
        self._stretch_ready = np.zeros(0, dtype=np.float32)
        return out.astype(np.float32)

    def parameter_schema(self) -> tuple[ParameterSpec, ...]:
        return SCHEMA

    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(changes_identity=False, requires_gpu=False)

    def latency_frames(self) -> int:
        if self._use_rubband:
            return int(self._shifter.get_start_delay())
        return int(self._shifter.outputLatency())

    def set_params(self, params: dict[str, Any], *, crossfade_ms: float = 0.0) -> None:
        pitch_changed = False
        formant_changed = False
        for key, value in params.items():
            if key not in _DEFAULTS:
                continue
            number = float(value)
            if self._params.get(key) == number:
                continue
            self._params[key] = number
            if key == "pitch_semitones":
                pitch_changed = True
            if key == "formant_semitones":
                formant_changed = True
        if pitch_changed or (formant_changed and self._use_rubband):
            self._stretch_pending = np.zeros(0, dtype=np.float32)
            self._stretch_ready = np.zeros(0, dtype=np.float32)
            self._rebuild_shifter()
        self._update_filters()
        if crossfade_ms > 0:
            self._fade_total = int(self.sample_rate * crossfade_ms / 1000.0)
            self._fade_left = self._fade_total

    def _update_filters(self) -> None:
        room = float(self._params["room"])
        self._reverb.room_size = 0.28 + 0.50 * room
        self._reverb.wet_level = 0.82 * room
        self._reverb.dry_level = max(0.22, 1.0 - 0.52 * room)

        body = float(self._params["formant_semitones"]) / 12.0
        self._body_low.gain_db = -7.0 * body
        self._body_high.gain_db = 5.0 * body

        tilt = (0.5 - float(self._params["tonality"])) * 2.0
        self._tone_low.gain_db = -6.0 * tilt
        self._tone_high.gain_db = 9.0 * tilt

        distance = float(self._params["distance"])
        cutoff = 12000.0 * (1500.0 / 12000.0) ** distance
        self._dist_lp1.cutoff_frequency_hz = cutoff
        self._dist_lp2.cutoff_frequency_hz = cutoff

    def _distance_gain(self) -> float:
        amount = float(self._params["distance"])
        if amount < 0.01:
            return 1.0
        return _db_to_gain(-10.0 * amount**1.5)

    def apply_macro(self, tag: str) -> None:
        macro = self._macro_defs.get(tag)
        if macro is not None:
            self.set_params(macro.params)

    def process_block(self, block: np.ndarray) -> np.ndarray:
        samples = np.asarray(block, dtype=np.float32)
        if samples.size != self.block_size:
            raise ValueError(
                f"process_block expected {self.block_size} samples, got {samples.size}"
            )
        work = samples.copy()
        work = self._apply_gate(work)
        pitch = float(self._params["pitch_semitones"])
        if self._use_rubband:
            self._shifter.shift_into(work, self._live_out)
            work = self._live_out.copy()
        elif abs(pitch) >= 0.05:
            work = self._stretch_block(work)
        if not self._use_rubband:
            work = self._apply_body(work)
        work = self._apply_tonality(work)
        work = self._apply_growl(work)
        work = self._apply_hollow(work)
        work = self._apply_breath(work)
        work = self._apply_distance(work)
        work = self._apply_room_block(work)
        peak = float(np.max(np.abs(work))) if work.size else 0.0
        if peak > 0.98:
            work = self._run(self._limiter, work)
        else:
            work = np.clip(work, -1.0, 1.0)
        work = self._crossfade(work)
        return np.ascontiguousarray(work, dtype=np.float32)

    def _crossfade(self, block: np.ndarray) -> np.ndarray:
        previous = self._fade_prev
        self._fade_prev = block.copy()
        if self._fade_left <= 0 or previous is None or self._fade_total <= 0:
            return block
        done = self._fade_total - self._fade_left
        ramp = np.linspace(
            done / self._fade_total,
            min(1.0, (done + block.size) / self._fade_total),
            block.size,
            dtype=np.float32,
        )
        mixed = previous * (1.0 - ramp) + block * ramp
        self._fade_left = max(0, self._fade_left - block.size)
        return mixed.astype(np.float32)

    def _follow(
        self,
        levels: np.ndarray,
        state: float,
        *,
        attack_ms: float,
        release_ms: float,
    ) -> tuple[np.ndarray, float]:
        """Attack/release follower over sub-frame levels."""
        frame_ms = 1000.0 * _ENV_FRAME / self.sample_rate
        attack = 1.0 - float(np.exp(-frame_ms / max(attack_ms, 1e-3)))
        release = 1.0 - float(np.exp(-frame_ms / max(release_ms, 1e-3)))
        out = np.empty(levels.size, dtype=np.float64)
        for i, level in enumerate(levels):
            coeff = attack if level > state else release
            state += coeff * (float(level) - state)
            out[i] = state
        return out, state

    def _apply_gate(self, work: np.ndarray) -> np.ndarray:
        amount = float(self._params["gate"])
        levels = _frame_rms(work, _ENV_FRAME)
        env, self._gate_env = self._follow(
            levels, self._gate_env, attack_ms=1.5, release_ms=60.0
        )
        if amount < 0.01:
            self._gate_gain = 1.0
            self._gate_open = True
            return work
        # Threshold on the speech RMS envelope: −54 dB (just above a quiet room)
        # up to −28 dB. Hysteresis + hold keep word tails from chattering.
        open_at = _db_to_gain(-54.0 + 26.0 * amount)
        close_at = open_at * _db_to_gain(-6.0)
        floor = _db_to_gain(-30.0 - 40.0 * amount)
        hold_frames = int(0.120 * self.sample_rate / _ENV_FRAME)
        frame_ms = 1000.0 * _ENV_FRAME / self.sample_rate
        open_step = 1.0 - float(np.exp(-frame_ms / 2.5))
        close_step = 1.0 - float(np.exp(-frame_ms / 70.0))
        gains = np.empty(env.size, dtype=np.float64)
        gain = self._gate_gain
        for i, level in enumerate(env):
            if level >= open_at:
                self._gate_open = True
                self._gate_hold = hold_frames
            elif level < close_at:
                if self._gate_hold > 0:
                    self._gate_hold -= 1
                else:
                    self._gate_open = False
            target = 1.0 if self._gate_open else floor
            step = open_step if target > gain else close_step
            gain += step * (target - gain)
            gains[i] = gain
        previous = self._gate_gain
        self._gate_gain = gain
        ramp = _per_sample(gains, previous, work.size, _ENV_FRAME)
        return (work * ramp).astype(np.float32)

    def _apply_body(self, work: np.ndarray) -> np.ndarray:
        # Streaming fallback only: a chest/thin shelf pair. Preview render does a
        # true formant shift in _stretch_offline and zeroes this parameter.
        body = float(self._params["formant_semitones"]) / 12.0
        if abs(body) < 0.02:
            return work
        wet = self._run(self._body_high, self._run(self._body_low, work))
        return (wet * _db_to_gain(1.2 * body)).astype(np.float32)

    def _apply_tonality(self, work: np.ndarray) -> np.ndarray:
        # 0 = brighter, 1 = darker. Default 0.5 is dry. Shelves tilt around 1 kHz;
        # the pair is roughly level-neutral so it reads as timbre, not volume.
        tilt = (0.5 - float(self._params["tonality"])) * 2.0
        if abs(tilt) < 0.02:
            return work
        wet = self._run(self._tone_high, self._run(self._tone_low, work))
        return (wet * _db_to_gain(-1.5 * tilt)).astype(np.float32)

    def _apply_growl(self, work: np.ndarray) -> np.ndarray:
        amount = float(self._params["growl"])
        if amount < 0.01:
            return work
        drive = 1.0 + 22.0 * amount
        bias = 0.35 * amount
        pre = self._run(self._growl_pre, work).astype(np.float64)
        shaped = np.tanh(drive * pre + bias) - np.tanh(bias)
        # Small signals see gain ≈ drive; take it back out so the saturation
        # reads as grit and compression rather than a louder voice.
        shaped /= drive**0.85
        n = work.size
        rate_hz = 34.0 + 18.0 * amount
        phase = (
            self._growl_phase + 2 * np.pi * rate_hz * np.arange(n) / self.sample_rate
        )
        self._growl_phase = float(
            (phase[-1] + 2 * np.pi * rate_hz / self.sample_rate) % (2 * np.pi)
        )
        depth = 0.45 * amount
        rasp = 1.0 - depth * (0.5 + 0.5 * np.sin(phase))
        wet = self._run(self._growl_post, (shaped * rasp).astype(np.float32))
        mix = amount**0.7
        return ((1.0 - mix) * work + mix * wet).astype(np.float32)

    def _apply_hollow(self, work: np.ndarray) -> np.ndarray:
        # Unipolar: 0 is off. Wet = a cupped, tube-like cavity (band-limited with
        # a mid resonance and a short boxy comb), not a ringing metallic comb.
        amount = float(self._params["hollow"])
        if amount < 0.01:
            return work
        wet = self._run(self._hollow_hp2, self._run(self._hollow_hp, work))
        wet = self._run(self._hollow_lp2, self._run(self._hollow_lp, wet))
        wet = self._run(self._hollow_peak, wet)
        wet = self._run(self._hollow_comb, wet)
        wet = wet * 1.35
        mix = amount
        return ((1.0 - mix) * work + mix * wet).astype(np.float32)

    def _apply_breath(self, work: np.ndarray) -> np.ndarray:
        """Streaming Breath: one whisperised frame per block (one block of lag).

        Frame = previous block + this block under a sqrt-Hann window, so the
        50 %-overlap add is flat. Render does the whole Take in ``breath_air``.
        """
        amount = float(self._params["breath"])
        if amount < 0.01:
            self._breath_prev = work.copy()
            self._breath_tail.fill(0.0)
            return work
        n = work.size
        frame = np.concatenate([self._breath_prev, work]).astype(np.float64)
        frame = frame * self._breath_window
        air = _whisper_frames(frame[None, :], self._breath_weight, self._rng)[0]
        air = air * self._breath_window
        out_air = air[:n] + self._breath_tail
        self._breath_tail = air[n:].copy()
        self._breath_prev = work.copy()
        return mix_breath(work, out_air.astype(np.float32), amount)

    def _apply_distance(self, work: np.ndarray) -> np.ndarray:
        amount = float(self._params["distance"])
        if amount < 0.01:
            return work
        darker = self._run(self._dist_lp2, self._run(self._dist_lp1, work))
        if self._defer_distance_gain:
            return darker
        return (darker * self._distance_gain()).astype(np.float32)

    def _apply_room_block(self, work: np.ndarray) -> np.ndarray:
        amount = float(self._params["room"])
        if amount < 0.01:
            return work
        delay = self._room_delay
        index = self._room_i
        n = delay.size
        rate = self.sample_rate
        t1 = max(1, int(rate * 0.019)) % n
        t2 = max(1, int(rate * 0.037)) % n
        t3 = max(1, int(rate * 0.058)) % n
        mix = 0.78 * amount
        out = np.empty_like(work)
        for i, sample in enumerate(work):
            value = float(sample)
            delay[index] = value
            echo = (
                0.62 * delay[(index - t1) % n]
                + 0.40 * delay[(index - t2) % n]
                + 0.26 * delay[(index - t3) % n]
            )
            out[i] = value + mix * float(echo)
            index = (index + 1) % n
        self._room_i = index
        return out.astype(np.float32)

    def _apply_room_offline(self, samples: np.ndarray, room: float) -> np.ndarray:
        """Whole-Take Pedalboard reverb — 256-sample grains never built a tail."""
        self._reverb.room_size = 0.28 + 0.50 * room
        self._reverb.damping = 0.30
        self._reverb.wet_level = 0.82 * room
        self._reverb.dry_level = max(0.22, 1.0 - 0.52 * room)
        silent = np.zeros(self.block_size, dtype=np.float32)
        self._reverb.process(
            silent, self.sample_rate, buffer_size=self.block_size, reset=True
        )
        buf = min(8192, max(self.block_size, int(samples.size)))
        return _plugin_mono(
            self._reverb.process(
                samples, self.sample_rate, buffer_size=buf, reset=False
            )
        )

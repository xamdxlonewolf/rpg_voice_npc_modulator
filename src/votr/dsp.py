# Copyright (C) 2026 Michael Cobb
# SPDX-License-Identifier: GPL-3.0-or-later

"""DSP Engine v1: gate → pitch/formant → texture → Pedalboard → limiter."""

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
    ParameterSpec("formant_semitones", "Body / formant", -12.0, 12.0, 0.0),
    ParameterSpec("tonality", "Tonality", 0.0, 1.0, 0.5),
    ParameterSpec("growl", "Growl", 0.0, 1.0, 0.0),
    ParameterSpec("hollow", "Hollow", 0.0, 1.0, 0.0),
    ParameterSpec("room", "Room", 0.0, 1.0, 0.0),
    ParameterSpec("distance", "Distance", 0.0, 1.0, 0.0),
    ParameterSpec("breath", "Breath", 0.0, 1.0, 0.0),
    ParameterSpec("gate", "Gate", 0.0, 1.0, 0.0),
)

_DEFAULTS = {spec.key: spec.default for spec in SCHEMA}
# python-stretch only transposes when the process hop is ≥ outputLatency (~30 ms).
STRETCH_HOP = 1440


def _plugin_mono(array: np.ndarray) -> np.ndarray:
    samples = np.asarray(array, dtype=np.float32)
    if samples.ndim == 2:
        samples = samples[0] if samples.shape[0] <= samples.shape[1] else samples[:, 0]
    return samples


def _semitone_ratio(semitones: float) -> float:
    return float(2.0 ** (semitones / 12.0))


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
        self._env = 0.0
        self._phase = 0.0
        self._comb = np.zeros(max(8, int(sample_rate * 0.006)), dtype=np.float32)
        self._comb_i = 0
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
        self._lp = 0.0
        self._tone_lp = 0.0
        self._init_effects()
        self._rebuild_shifter()

    def _init_effects(self) -> None:
        import pedalboard as pb

        self._lowpass = pb.LowpassFilter(cutoff_frequency_hz=12_000)
        self._reverb = pb.Reverb(
            room_size=0.25, damping=0.4, wet_level=0.0, dry_level=1.0, width=0.6
        )
        self._limiter = pb.Limiter(threshold_db=-1.5, release_ms=50)

    def reset(self) -> None:
        """Clear streaming leftovers before rendering a Take."""
        self._env = 0.0
        self._phase = 0.0
        self._comb.fill(0)
        self._comb_i = 0
        self._fade_left = 0
        self._fade_prev = None
        self._lp = 0.0
        self._tone_lp = 0.0
        self._stretch_pending = np.zeros(0, dtype=np.float32)
        self._stretch_ready = np.zeros(0, dtype=np.float32)
        self._rebuild_shifter()
        silent = np.zeros(self.block_size, dtype=np.float32)
        for plugin in (self._lowpass, self._reverb, self._limiter):
            plugin.process(
                silent, self.sample_rate, buffer_size=self.block_size, reset=True
            )

    def _stretch_offline(self, samples: np.ndarray) -> np.ndarray:
        self._rebuild_shifter()
        framed = np.ascontiguousarray(samples.reshape(1, -1))
        shifted = self._shifter.process(framed)
        out = np.asarray(shifted, dtype=np.float32).reshape(-1)
        if out.size >= samples.size:
            return out[: samples.size]
        return np.pad(out, (0, samples.size - out.size))

    def render(self, take: np.ndarray) -> np.ndarray:
        """Render a Take from a clean state — not leftover stream memory."""
        self.reset()
        samples = np.asarray(take, dtype=np.float32).reshape(-1)
        if samples.size == 0:
            return samples.copy()
        pitch = float(self._params["pitch_semitones"])
        if not self._use_rubband and abs(pitch) >= 0.05:
            samples = self._stretch_offline(samples)
            saved = pitch
            self._params["pitch_semitones"] = 0.0
            try:
                return super().render(samples)
            finally:
                self._params["pitch_semitones"] = saved
        return super().render(samples)

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
        out = np.concatenate(
            [np.zeros(pad, dtype=np.float32), self._stretch_ready]
        )
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
        room = float(self._params["room"])
        self._reverb.wet_level = 0.45 * room
        self._reverb.dry_level = max(0.2, 1.0 - 0.45 * room)
        distance = float(self._params["distance"])
        self._lowpass.cutoff_frequency_hz = 12_000 - distance * 10_800
        if crossfade_ms > 0:
            self._fade_total = int(self.sample_rate * crossfade_ms / 1000.0)
            self._fade_left = self._fade_total

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
        if float(self._params["distance"]) > 0.01:
            work = _plugin_mono(
                self._lowpass.process(
                    work, self.sample_rate, buffer_size=self.block_size, reset=False
                )
            )
            work *= 1.0 - 0.55 * float(self._params["distance"])
        if float(self._params["room"]) > 0.01:
            work = _plugin_mono(
                self._reverb.process(
                    work, self.sample_rate, buffer_size=self.block_size, reset=False
                )
            )
        peak = float(np.max(np.abs(work))) if work.size else 0.0
        need_limit = peak > 0.98 or float(self._params["growl"]) > 0.05
        if need_limit:
            work = _plugin_mono(
                self._limiter.process(
                    work, self.sample_rate, buffer_size=self.block_size, reset=False
                )
            )
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

    def _apply_gate(self, work: np.ndarray) -> np.ndarray:
        amount = float(self._params["gate"])
        rms = float(np.sqrt(np.mean(work * work)) + 1e-8)
        self._env = 0.85 * self._env + 0.15 * rms
        if amount < 0.01:
            return work
        threshold = 0.002 + amount * 0.08
        if self._env < threshold:
            work = work * (self._env / threshold)
        return work

    def _tilt(self, work: np.ndarray, amount: float, state_attr: str) -> np.ndarray:
        """One-pole tilt. Positive amount = brighter / smaller body."""
        if abs(amount) < 0.02:
            return work
        coeff = 1.0 - np.exp(-2.0 * np.pi * 350.0 / self.sample_rate)
        low = float(getattr(self, state_attr))
        out = np.empty_like(work)
        for i, sample in enumerate(work):
            low += coeff * (float(sample) - low)
            out[i] = sample + amount * (sample - low)
        setattr(self, state_attr, low)
        return out.astype(np.float32)

    def _apply_body(self, work: np.ndarray) -> np.ndarray:
        # Stretch has no formant API. Map Body / formant to a gentle tilt:
        # +semitones = smaller/thinner, −semitones = larger/darker.
        amount = float(self._params["formant_semitones"]) / 12.0 * 0.55
        return self._tilt(work, amount, "_lp")

    def _apply_tonality(self, work: np.ndarray) -> np.ndarray:
        amount = (0.5 - float(self._params["tonality"])) * 0.35
        return self._tilt(work, amount, "_tone_lp")

    def _apply_growl(self, work: np.ndarray) -> np.ndarray:
        amount = float(self._params["growl"])
        if amount < 0.01:
            return work
        drive = 1.0 + 3.0 * amount
        wet = np.tanh(work * drive).astype(np.float32) / np.tanh(drive)
        return ((1.0 - amount) * work + amount * wet).astype(np.float32)

    def _apply_hollow(self, work: np.ndarray) -> np.ndarray:
        amount = float(self._params["hollow"])
        if amount < 0.01:
            return work
        n = work.size
        phase = self._phase + 2 * np.pi * 90.0 * np.arange(n) / self.sample_rate
        step = 2 * np.pi * 90.0 / self.sample_rate
        self._phase = float((phase[-1] + step) % (2 * np.pi))
        ring = work * np.sin(phase).astype(np.float32)
        comb = np.empty_like(work)
        delay = self._comb
        index = self._comb_i
        for i, sample in enumerate(work):
            comb[i] = sample + 0.55 * delay[index]
            delay[index] = sample
            index = (index + 1) % delay.size
        self._comb_i = index
        wet = 0.15 * ring + 0.85 * comb
        mix = 0.55 * amount
        return ((1.0 - mix) * work + mix * wet).astype(np.float32)

    def _apply_breath(self, work: np.ndarray) -> np.ndarray:
        amount = float(self._params["breath"])
        if amount < 0.01:
            return work
        noise = self._rng.standard_normal(work.size).astype(np.float32)
        env = np.minimum(1.0, np.abs(work) * 4.0)
        return work + noise * env * (0.08 * amount)

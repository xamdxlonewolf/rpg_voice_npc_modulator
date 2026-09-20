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
    ParameterSpec("formant_semitones", "Formant (semitones)", -12.0, 12.0, 0.0),
    ParameterSpec("tonality", "Tonality", 0.0, 1.0, 0.5),
    ParameterSpec("growl", "Growl", 0.0, 1.0, 0.0),
    ParameterSpec("hollow", "Hollow", 0.0, 1.0, 0.0),
    ParameterSpec("room", "Room", 0.0, 1.0, 0.0),
    ParameterSpec("distance", "Distance", 0.0, 1.0, 0.0),
    ParameterSpec("breath", "Breath", 0.0, 1.0, 0.0),
    ParameterSpec("gate", "Gate", 0.0, 1.0, 0.0),
)

_DEFAULTS = {spec.key: spec.default for spec in SCHEMA}


def _plugin_mono(array: np.ndarray) -> np.ndarray:
    samples = np.asarray(array, dtype=np.float32)
    if samples.ndim == 2:
        samples = samples[0] if samples.shape[0] <= samples.shape[1] else samples[:, 0]
    return samples


def _semitone_ratio(semitones: float) -> float:
    return float(2.0 ** (semitones / 12.0))


def _formant_shift(block: np.ndarray, semitones: float) -> np.ndarray:
    """Block-safe spectral-envelope shift used when LiveShifter is absent."""
    if abs(semitones) < 0.05:
        return block
    spectrum = np.fft.rfft(block)
    magnitude = np.abs(spectrum)
    phase = np.angle(spectrum)
    ratio = _semitone_ratio(semitones)
    source = np.arange(magnitude.size) / ratio
    shifted = np.interp(
        np.arange(magnitude.size), source, magnitude, left=0.0, right=0.0
    )
    restored = np.fft.irfft(shifted * np.exp(1j * phase), n=block.size)
    return restored.astype(np.float32)


class DspEngine(BaseEngine):
    """CPU Engine that modulates the GM's own voice (ADR-0004)."""

    engine_id = DSP_ENGINE_ID

    def __init__(
        self,
        *,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        macros: dict[str, Macro] | None = None,
        prefer_rubband: bool = True,
    ) -> None:
        self._use_rubband = bool(prefer_rubband and rubband_available())
        if not self._use_rubband and not stretch_available():
            raise RuntimeError("DSP Engine needs python-stretch or rubband")
        block_size = 512 if self._use_rubband else 256
        super().__init__(block_size=block_size, sample_rate=sample_rate)
        self._params = dict(_DEFAULTS)
        self._macro_defs = macros or {}
        self._env = 0.0
        self._phase = 0.0
        self._comb = np.zeros(max(8, int(sample_rate * 0.006)), dtype=np.float32)
        self._comb_i = 0
        self._shifter: Any = None
        self._stretch_in = np.zeros((1, block_size), dtype=np.float32)
        self._live_out = np.zeros(block_size, dtype=np.float32)
        self._rng = np.random.default_rng()
        self._init_effects()
        self._rebuild_shifter()

    def _init_effects(self) -> None:
        import pedalboard as pb

        self._lowpass = pb.LowpassFilter(cutoff_frequency_hz=12_000)
        self._reverb = pb.Reverb(
            room_size=0.25, damping=0.4, wet_level=0.0, dry_level=1.0, width=0.6
        )
        self._limiter = pb.Limiter(threshold_db=-1.5, release_ms=50)

    def _rebuild_shifter(self) -> None:
        pitch = float(self._params["pitch_semitones"])
        formant = float(self._params["formant_semitones"])
        tonality = float(self._params["tonality"])
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
        limit = (2000 + (1.0 - tonality) * 6000) / self.sample_rate
        self._shifter.setTransposeSemitones(pitch, limit)

    def parameter_schema(self) -> tuple[ParameterSpec, ...]:
        return SCHEMA

    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(changes_identity=False, requires_gpu=False)

    def latency_frames(self) -> int:
        if self._use_rubband:
            return int(self._shifter.get_start_delay())
        return int(self._shifter.outputLatency())

    def set_params(self, params: dict[str, Any]) -> None:
        changed = False
        for key, value in params.items():
            if key not in _DEFAULTS:
                continue
            number = float(value)
            if self._params.get(key) != number:
                self._params[key] = number
                changed = True
        if changed:
            self._rebuild_shifter()
            room = float(self._params["room"])
            self._reverb.wet_level = 0.45 * room
            self._reverb.dry_level = max(0.2, 1.0 - 0.45 * room)
            distance = float(self._params["distance"])
            self._lowpass.cutoff_frequency_hz = 12_000 - distance * 10_800

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
        if not self._use_rubband:
            work = _formant_shift(work, float(self._params["formant_semitones"]))
            self._stretch_in[0] = work
            work = self._shifter.process(self._stretch_in)[0].astype(np.float32)
        else:
            self._shifter.shift_into(work, self._live_out)
            work = self._live_out.copy()
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
        limited = _plugin_mono(
            self._limiter.process(
                work, self.sample_rate, buffer_size=self.block_size, reset=False
            )
        )
        return np.ascontiguousarray(limited, dtype=np.float32)

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

    def _apply_growl(self, work: np.ndarray) -> np.ndarray:
        amount = float(self._params["growl"])
        if amount < 0.01:
            return work
        drive = 1.0 + 8.0 * amount
        return np.tanh(work * drive).astype(np.float32) / np.tanh(drive)

    def _apply_hollow(self, work: np.ndarray) -> np.ndarray:
        amount = float(self._params["hollow"])
        if amount < 0.01:
            return work
        n = work.size
        phase = self._phase + 2 * np.pi * 170.0 * np.arange(n) / self.sample_rate
        step = 2 * np.pi * 170.0 / self.sample_rate
        self._phase = float((phase[-1] + step) % (2 * np.pi))
        ring = work * np.sin(phase).astype(np.float32)
        comb = np.empty_like(work)
        delay = self._comb
        index = self._comb_i
        for i, sample in enumerate(work):
            comb[i] = sample + 0.65 * delay[index]
            delay[index] = sample
            index = (index + 1) % delay.size
        self._comb_i = index
        wet = 0.5 * ring + 0.5 * comb
        return ((1.0 - amount) * work + amount * wet).astype(np.float32)

    def _apply_breath(self, work: np.ndarray) -> np.ndarray:
        amount = float(self._params["breath"])
        if amount < 0.01:
            return work
        noise = self._rng.standard_normal(work.size).astype(np.float32)
        env = np.minimum(1.0, np.abs(work) * 4.0)
        return work + noise * env * (0.08 * amount)

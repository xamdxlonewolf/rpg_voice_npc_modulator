# Copyright (C) 2026 Michael Cobb
# SPDX-License-Identifier: GPL-3.0-or-later

"""Neural Engine: zero-shot voice conversion behind the Engine seam (ADR-0004).

The Engine owns everything that is not the model: 48 kHz ↔ model-rate
resampling, the streaming window scheme (history | current | smooth | future,
as in X-VC's ``run_streaming``), crossfading, dry/wet mix with a matched dry
delay, and ``render`` that pushes a Take through the very same ``process_block``
path so Preview equals live (ADR-0005). The model itself sits behind the small
``VoiceConverter`` protocol so a fake can stand in where there is no GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from votr.dsp import active_rms
from votr.engine import (
    DEFAULT_SAMPLE_RATE,
    BaseEngine,
    EngineCapabilities,
    ParameterSpec,
)

NEURAL_ENGINE_ID = "neural-v0"
REFERENCE_CLIP_KEY = "reference_clip"

SCHEMA: tuple[ParameterSpec, ...] = (
    ParameterSpec("mix", "Mix (0 = your voice, 1 = converted)", 0.0, 1.0, 1.0),
    ParameterSpec(
        "quality",
        "Quality vs speed (0 = speed, 1 = balanced, 2 = quality)",
        0.0,
        2.0,
        1.0,
    ),
)
_DEFAULTS = {spec.key: spec.default for spec in SCHEMA}


class NeuralUnavailable(RuntimeError):
    """The Neural Engine cannot run here; the caller should fall back to DSP."""


class VoiceConverter(Protocol):
    """A zero-shot voice conversion model at its own sample rate."""

    sample_rate: int

    def set_reference(self, clip: np.ndarray, sample_rate: int) -> None:
        """Register the target voice from a mono float32 reference clip."""

    def convert_window(self, window: np.ndarray) -> np.ndarray:
        """Convert one model-rate window; returns the same number of samples."""


_PEAK_CEILING = 0.97


@dataclass(frozen=True)
class StreamWindow:
    """X-VC streaming geometry in milliseconds (see bins/infer_utils.py)."""

    chunk_ms: int = 2400
    current_ms: int = 240
    future_ms: int = 100
    smooth_ms: int = 20

    @property
    def history_ms(self) -> int:
        return self.chunk_ms - self.current_ms - self.smooth_ms - self.future_ms

    def validate(self) -> None:
        if self.current_ms <= 0:
            raise ValueError("current_ms must be > 0")
        if self.history_ms < 0:
            raise ValueError("chunk must cover current + smooth + future")


# X-VC is one-step codec conversion: no diffusion steps, no guidance scale.
# The real knobs are the streaming window (chunk stays 2400 ms to match training).
# Speed = paper streaming (120 ms current + 20 ms overlap + 100 ms future).
# Quality = fewer joins and more lookahead, higher convert latency.
QUALITY_WINDOWS: tuple[StreamWindow, ...] = (
    StreamWindow(chunk_ms=2400, current_ms=120, future_ms=100, smooth_ms=20),
    StreamWindow(chunk_ms=2400, current_ms=240, future_ms=100, smooth_ms=20),
    StreamWindow(chunk_ms=2400, current_ms=480, future_ms=200, smooth_ms=40),
)


def quality_index(value: float) -> int:
    return int(max(0, min(len(QUALITY_WINDOWS) - 1, round(float(value)))))


def window_for_quality(value: float) -> StreamWindow:
    return QUALITY_WINDOWS[quality_index(value)]


def _lowpass_fir(cutoff: float, taps: int) -> np.ndarray:
    """Windowed-sinc lowpass; ``cutoff`` as a fraction of the sample rate."""
    n = np.arange(taps) - (taps - 1) / 2.0
    kernel = 2.0 * cutoff * np.sinc(2.0 * cutoff * n)
    kernel *= np.blackman(taps)
    return (kernel / np.sum(kernel)).astype(np.float64)


class _Resampler:
    """Streaming rational resampler (zero-stuff → FIR → decimate) with state."""

    def __init__(self, up: int, down: int, taps: int = 96) -> None:
        self.up = int(up)
        self.down = int(down)
        cutoff = 0.5 / max(self.up, self.down)
        self._kernel = _lowpass_fir(cutoff, taps) * self.up
        self._history = np.zeros(taps - 1, dtype=np.float64)
        self._phase = 0

    @property
    def delay_in_samples(self) -> float:
        """Group delay in *input* samples."""
        return (self._kernel.size - 1) / 2.0 / self.up

    def reset(self) -> None:
        self._history.fill(0.0)
        self._phase = 0

    def process(self, block: np.ndarray) -> np.ndarray:
        if block.size == 0:
            return np.zeros(0, dtype=np.float32)
        stuffed = np.zeros(block.size * self.up, dtype=np.float64)
        stuffed[:: self.up] = block
        signal = np.concatenate([self._history, stuffed])
        filtered = np.convolve(signal, self._kernel, mode="valid")
        self._history = signal[-(self._kernel.size - 1) :]
        picked = filtered[self._phase :: self.down]
        consumed = filtered.size
        self._phase = (self._phase - consumed) % self.down
        return picked.astype(np.float32)


class NeuralEngine(BaseEngine):
    """Engine that makes the GM sound like the registered reference voice."""

    engine_id = NEURAL_ENGINE_ID

    def __init__(
        self,
        converter: VoiceConverter,
        *,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        block_size: int = 512,
        window: StreamWindow = StreamWindow(),
    ) -> None:
        window.validate()
        super().__init__(block_size=int(block_size), sample_rate=sample_rate)
        self._converter = converter
        self._window = window
        model_rate = int(converter.sample_rate)
        if sample_rate < model_rate or sample_rate % model_rate:
            raise NeuralUnavailable(
                f"host rate {sample_rate} must be an integer multiple of the "
                f"model rate {model_rate}"
            )
        self._model_rate = model_rate
        factor = sample_rate // model_rate
        self._down = _Resampler(1, factor)
        self._up = _Resampler(factor, 1)
        self._params: dict[str, Any] = dict(_DEFAULTS)
        self._reference_path = ""
        self._has_reference = False
        self._apply_window(window)
        self._reset_streams()

    def _apply_window(self, window: StreamWindow) -> None:
        window.validate()
        self._window = window
        ms = self._model_rate / 1000.0
        self._cur = int(window.current_ms * ms)
        self._fut = int(window.future_ms * ms)
        self._smooth = int(window.smooth_ms * ms)
        self._hist = int(window.history_ms * ms)
        self._latency = self._cur + self._smooth + self._fut

    # -- Engine protocol -----------------------------------------------------

    def parameter_schema(self) -> tuple[ParameterSpec, ...]:
        return SCHEMA

    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(changes_identity=True, requires_gpu=True)

    def latency_frames(self) -> int:
        ratio = self.sample_rate / self._model_rate
        fir = self._down.delay_in_samples + self._up.delay_in_samples * ratio
        return int(round(self._latency * ratio + fir))

    @property
    def has_reference(self) -> bool:
        return self._has_reference

    @property
    def reference_path(self) -> str:
        return self._reference_path

    def set_params(self, params: dict[str, Any], **_ignored: Any) -> None:
        quality_changed = False
        for key, value in params.items():
            if key == REFERENCE_CLIP_KEY:
                self.set_reference_clip(str(value or ""))
            elif key == "quality":
                index = quality_index(value)
                self._params["quality"] = float(index)
                wanted = QUALITY_WINDOWS[index]
                if wanted != self._window:
                    self._apply_window(wanted)
                    quality_changed = True
            elif key == "mix":
                self._params["mix"] = min(1.0, max(0.0, float(value)))
            elif key in _DEFAULTS:
                self._params[key] = float(value)
        if quality_changed:
            self.reset()

    def params(self) -> dict[str, Any]:
        merged = dict(self._params)
        merged[REFERENCE_CLIP_KEY] = self._reference_path
        return merged

    def apply_macro(self, tag: str) -> None:
        return None

    def set_reference_clip(self, path: str) -> None:
        """Load a mono WAV reference clip and hand it to the converter."""
        if not path:
            self._reference_path = ""
            self._has_reference = False
            return
        if path == self._reference_path and self._has_reference:
            return
        from votr.wavutil import read_wav

        samples, rate = read_wav(Path(path))
        clip = np.asarray(samples, dtype=np.float32).reshape(-1)
        self._converter.set_reference(clip, int(rate))
        self._reference_path = path
        self._has_reference = True

    # -- streaming ----------------------------------------------------------

    def _reset_streams(self) -> None:
        self._down.reset()
        self._up.reset()
        self._source = np.zeros(0, dtype=np.float32)
        self._source_offset = 0
        self._emitted = 0
        self._tail = np.zeros(self._smooth, dtype=np.float32)
        self._first_chunk = True
        self._out_model = np.zeros(0, dtype=np.float32)
        # Prime the output with exactly the model latency so the first chunk
        # lands on time and latency_frames() is exact, not block-quantised.
        ratio = self.sample_rate // self._model_rate
        self._out_host = np.zeros(self._latency * ratio, dtype=np.float32)
        self._dry = np.zeros(self.latency_frames(), dtype=np.float32)

    def reset(self) -> None:
        self._reset_streams()

    def _window_at(self, start: int, end: int) -> np.ndarray:
        """Model-rate source samples [start, end) with zero padding outside."""
        lo = start - self._source_offset
        hi = end - self._source_offset
        left = max(0, -lo)
        right = max(0, hi - self._source.size)
        body = self._source[max(0, lo) : max(0, min(hi, self._source.size))]
        return np.concatenate(
            [np.zeros(left, np.float32), body, np.zeros(right, np.float32)]
        )

    def _run_ready_chunks(self) -> None:
        available = self._source_offset + self._source.size
        while available >= self._emitted + self._cur + self._smooth + self._fut:
            start = self._emitted - self._hist
            end = self._emitted + self._cur + self._smooth + self._fut
            window = self._window_at(start, end)
            out = np.asarray(self._converter.convert_window(window), dtype=np.float32)
            out = out.reshape(-1)
            if out.size < window.size:
                out = np.pad(out, (0, window.size - out.size))
            out = out[: window.size]
            chunk = out[self._hist : self._hist + self._cur].copy()
            if self._smooth:
                if not self._first_chunk:
                    fade_in = 0.5 * (
                        1 - np.cos(np.pi * np.linspace(0, 1, self._smooth))
                    )
                    head = chunk[: self._smooth]
                    chunk[: self._smooth] = self._tail * (1 - fade_in) + head * fade_in
                tail_start = self._hist + self._cur
                self._tail = out[tail_start : tail_start + self._smooth].copy()
            self._first_chunk = False
            self._out_model = np.concatenate([self._out_model, chunk])
            self._emitted += self._cur
            keep_from = self._emitted - self._hist
            drop = keep_from - self._source_offset
            if drop > 0:
                self._source = self._source[drop:]
                self._source_offset += drop

    def process_block(self, block: np.ndarray) -> np.ndarray:
        samples = np.asarray(block, dtype=np.float32).reshape(-1)
        if samples.size != self.block_size:
            raise ValueError(
                f"process_block expected {self.block_size} samples, got {samples.size}"
            )
        mix = float(self._params["mix"])
        if not self._has_reference or mix <= 0.0:
            return samples.copy()
        self._source = np.concatenate([self._source, self._down.process(samples)])
        self._run_ready_chunks()
        if self._out_model.size:
            self._out_host = np.concatenate(
                [self._out_host, self._up.process(self._out_model)]
            )
            self._out_model = np.zeros(0, dtype=np.float32)
        wet = np.zeros(self.block_size, dtype=np.float32)
        take = min(self.block_size, self._out_host.size)
        if take:
            wet[:take] = self._out_host[:take]
            self._out_host = self._out_host[take:]
        if mix >= 1.0:
            return np.clip(wet, -1.0, 1.0)
        # Dry is delayed by the same latency so the two do not smear.
        self._dry = np.concatenate([self._dry, samples])
        dry = self._dry[: self.block_size]
        self._dry = self._dry[self.block_size :]
        return np.clip((1.0 - mix) * dry + mix * wet, -1.0, 1.0).astype(np.float32)

    def render(self, take: np.ndarray) -> np.ndarray:
        """Same block path as live, then latency trimmed and level matched."""
        self.reset()
        samples = np.asarray(take, dtype=np.float32).reshape(-1)
        if samples.size == 0:
            return samples.copy()
        if not self._has_reference or float(self._params["mix"]) <= 0.0:
            return samples.copy()
        latency = self.latency_frames()
        # Condition the whole Take the way the model expects (X-VC's
        # volume_normalize — that is model input, not a volume knob). Then
        # restore the Take's own level so a quiet mic stays quiet unless the
        # GM raised Mic gain. No extra silent boost on top.
        normalize = getattr(self._converter, "normalize", None)
        source = normalize(samples) if callable(normalize) else samples
        padded = np.concatenate(
            [source, np.zeros(latency + self.block_size, dtype=np.float32)]
        )
        out = super().render(padded)
        out = out[latency : latency + samples.size]
        dry_level = active_rms(samples, self.sample_rate)
        wet_level = active_rms(out, self.sample_rate)
        if dry_level > 1e-6 and wet_level > 1e-6:
            out = out * min(8.0, max(0.125, dry_level / wet_level))
        # Never hard-clip: bring the peak under the ceiling instead of cracking.
        peak = float(np.max(np.abs(out))) if out.size else 0.0
        if peak > _PEAK_CEILING:
            out = out * (_PEAK_CEILING / peak)
        return out.astype(np.float32)

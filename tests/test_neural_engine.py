# Copyright (C) 2026 Michael Cobb
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from votr.engine import BaseEngine, Engine
from votr.neural_engine import (
    NEURAL_ENGINE_ID,
    REFERENCE_CLIP_KEY,
    NeuralEngine,
    NeuralUnavailable,
    StreamWindow,
    _Resampler,
)
from votr.wavutil import write_wav

RATE = 48_000


class RecordingConverter:
    """Identity with a sign flip so converted audio is provably 'converted'."""

    sample_rate = 16_000

    def __init__(self) -> None:
        self.windows: list[int] = []
        self.reference: tuple[int, int] | None = None

    def set_reference(self, clip: np.ndarray, sample_rate: int) -> None:
        self.reference = (clip.size, sample_rate)

    def convert_window(self, window: np.ndarray) -> np.ndarray:
        self.windows.append(window.size)
        return -window


def _speech(seconds: float) -> np.ndarray:
    t = np.arange(int(RATE * seconds)) / RATE
    tone = sum(0.1 / k * np.sin(2 * np.pi * 140 * k * t) for k in range(1, 6))
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 3.0 * t - np.pi / 2)
    return (tone * envelope).astype(np.float32)


def _engine(tmp_path: Path, **kwargs) -> tuple[NeuralEngine, RecordingConverter]:
    converter = RecordingConverter()
    engine = NeuralEngine(converter, **kwargs)
    clip = tmp_path / "ref.wav"
    write_wav(clip, _speech(0.5), RATE)
    engine.set_params({REFERENCE_CLIP_KEY: str(clip)})
    return engine, converter


def test_resampler_round_trip_is_transparent() -> None:
    t = np.arange(RATE) / RATE
    x = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    down, up = _Resampler(1, 3), _Resampler(3, 1)
    pieces = [
        up.process(down.process(x[i : i + 512])) for i in range(0, RATE - 512, 512)
    ]
    y = np.concatenate(pieces)
    lag = int(round(down.delay_in_samples + up.delay_in_samples * 3))
    n = min(y.size - lag, x.size)
    assert float(np.corrcoef(x[:n], y[lag : lag + n])[0, 1]) > 0.9999
    assert np.sqrt(np.mean(y[lag : lag + n] ** 2)) == pytest.approx(
        np.sqrt(np.mean(x[:n] ** 2)), rel=0.01
    )
    assert down.process(x[:300]).size == 100
    assert up.process(x[:100]).size == 300


def test_engine_declares_itself_and_rejects_odd_rates(tmp_path: Path) -> None:
    engine, converter = _engine(tmp_path)
    assert isinstance(engine, Engine)
    assert engine.engine_id == NEURAL_ENGINE_ID
    caps = engine.capabilities()
    assert caps.changes_identity and caps.requires_gpu
    assert [spec.key for spec in engine.parameter_schema()] == ["mix"]
    assert engine.params()["mix"] == 1.0
    assert engine.params()[REFERENCE_CLIP_KEY].endswith("ref.wav")
    assert converter.reference == (24_000, RATE)
    assert engine.has_reference
    with pytest.raises(NeuralUnavailable):
        NeuralEngine(RecordingConverter(), sample_rate=44_100)
    with pytest.raises(ValueError):
        StreamWindow(chunk_ms=300, current_ms=240, future_ms=100).validate()


def test_latency_matches_window_and_is_reported(tmp_path: Path) -> None:
    engine, _ = _engine(tmp_path, window=StreamWindow(2400, 240, 100, 20))
    expected_model_ms = 240 + 100 + 20
    assert engine.latency_frames() == pytest.approx(expected_model_ms * 48, abs=200)
    assert 300 <= engine.latency_frames() / 48 <= 500


@pytest.mark.parametrize("block_size", [256, 512, 1024])
def test_render_is_aligned_and_uses_streaming_windows(
    tmp_path: Path, block_size: int
) -> None:
    engine, converter = _engine(tmp_path, block_size=block_size)
    take = _speech(1.0)
    out = engine.render(take)
    assert out.shape == take.shape and out.dtype == np.float32
    # Converted (sign-flipped) and time-aligned with the input: no latency smear.
    assert float(np.corrcoef(take, -out)[0, 1]) > 0.999
    corr = np.correlate(out[:24000], -take[:24000], "full")
    assert int(np.argmax(corr)) - (24000 - 1) == 0
    # Every model call saw a full X-VC window at 16 kHz.
    assert converter.windows and all(size == 2400 * 16 for size in converter.windows)
    assert len(converter.windows) == int(np.ceil(1.0 / 0.24))


def test_render_equals_live_block_path(tmp_path: Path) -> None:
    engine, _ = _engine(tmp_path, block_size=512)
    take = _speech(0.8)
    rendered = engine.render(take)
    engine.reset()
    latency = engine.latency_frames()
    padded = np.concatenate([take, np.zeros(latency + 512, dtype=np.float32)])
    live = BaseEngine.render(engine, padded)[latency : latency + take.size]
    # render() only adds a whole-Take level match on top of the block path.
    gain = np.sqrt(np.mean(rendered**2)) / np.sqrt(np.mean(live**2))
    np.testing.assert_allclose(rendered, live * gain, atol=1e-5)


def test_mix_blends_a_latency_matched_dry_signal(tmp_path: Path) -> None:
    engine, _ = _engine(tmp_path)
    take = _speech(0.8)
    engine.set_params({"mix": 0.5})
    out = engine.render(take)
    # Wet is the exact negative of dry; a time-aligned 50/50 mix cancels.
    assert np.sqrt(np.mean(out**2)) < 0.05 * np.sqrt(np.mean(take**2))
    engine.set_params({"mix": 0.0})
    np.testing.assert_allclose(engine.render(take), take)


def test_without_reference_the_engine_passes_audio_through(tmp_path: Path) -> None:
    engine = NeuralEngine(RecordingConverter())
    assert not engine.has_reference
    take = _speech(0.3)
    np.testing.assert_allclose(engine.render(take), take)
    block = take[: engine.block_size]
    np.testing.assert_allclose(engine.process_block(block), block)
    with pytest.raises(ValueError):
        engine.process_block(take[:10])
    engine.set_params({REFERENCE_CLIP_KEY: ""})
    assert engine.params()[REFERENCE_CLIP_KEY] == ""

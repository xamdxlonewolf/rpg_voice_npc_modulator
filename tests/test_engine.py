# Copyright (C) 2026 Michael Cobb
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest

from votr.engine import Engine, EngineCapabilities, PassthroughEngine


def _write_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    pcm = np.clip(np.rint(samples * 32767.0), -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())


def _read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wav:
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        sample_rate = wav.getframerate()
        raw = wav.readframes(wav.getnframes())
    pcm = np.frombuffer(raw, dtype="<i2")
    return pcm.astype(np.float32) / 32767.0, sample_rate


def _known_take(path: Path, sample_rate: int, n_samples: int = 1000) -> np.ndarray:
    """Write a known ramp WAV and return the samples as loaded from disk."""
    pcm = np.linspace(-20000, 20000, n_samples, dtype=np.int16)
    samples = pcm.astype(np.float32) / 32767.0
    _write_wav(path, samples, sample_rate)
    loaded, loaded_rate = _read_wav(path)
    assert loaded_rate == sample_rate
    np.testing.assert_array_equal(loaded, samples)
    return loaded


def _concatenated_process_block(
    engine: PassthroughEngine, take: np.ndarray
) -> np.ndarray:
    remainder = take.size % engine.block_size
    if remainder:
        padded = np.concatenate(
            [take, np.zeros(engine.block_size - remainder, dtype=np.float32)]
        )
    else:
        padded = take
    chunks = [
        engine.process_block(padded[i : i + engine.block_size])
        for i in range(0, padded.size, engine.block_size)
    ]
    return np.concatenate(chunks)[: take.size]


def test_passthrough_satisfies_engine_protocol() -> None:
    assert isinstance(PassthroughEngine(), Engine)


def test_passthrough_is_identity(tmp_path: Path) -> None:
    engine = PassthroughEngine(block_size=256, sample_rate=48_000)
    take = _known_take(tmp_path / "known.wav", engine.sample_rate)
    np.testing.assert_array_equal(engine.render(take), take)


def test_render_equals_concatenated_process_block(tmp_path: Path) -> None:
    engine = PassthroughEngine(block_size=256, sample_rate=48_000)
    take = _known_take(tmp_path / "known.wav", engine.sample_rate, n_samples=1000)
    rendered = engine.render(take)
    concatenated = _concatenated_process_block(engine, take)
    np.testing.assert_array_equal(rendered, concatenated)


def test_default_render_loops_non_identity_process_block(tmp_path: Path) -> None:
    class GainEngine(PassthroughEngine):
        def process_block(self, block: np.ndarray) -> np.ndarray:
            return super().process_block(block) * 2.0

    engine = GainEngine(block_size=256, sample_rate=48_000)
    take = _known_take(tmp_path / "known.wav", engine.sample_rate, n_samples=1000)
    rendered = engine.render(take)
    np.testing.assert_array_equal(rendered, _concatenated_process_block(engine, take))
    np.testing.assert_allclose(rendered, take * 2.0, atol=1e-6)


def test_process_block_rejects_wrong_length() -> None:
    engine = PassthroughEngine(block_size=256)
    with pytest.raises(ValueError, match="expected 256"):
        engine.process_block(np.zeros(128, dtype=np.float32))


def test_capabilities_are_cpu_identity_preserving() -> None:
    engine = PassthroughEngine()
    assert engine.capabilities() == EngineCapabilities(
        changes_identity=False,
        requires_gpu=False,
    )
    assert engine.parameter_schema() == ()


def test_set_params_and_apply_macro_do_not_change_audio() -> None:
    engine = PassthroughEngine(macros={"gravelly": {"unused": 1.0}})
    dry = np.linspace(-0.5, 0.5, engine.block_size, dtype=np.float32)
    engine.set_params({"unused": 0.25})
    engine.apply_macro("gravelly")
    engine.apply_macro("unknown-tag")
    np.testing.assert_array_equal(engine.process_block(dry), dry)
    assert engine.params()["unused"] == 1.0

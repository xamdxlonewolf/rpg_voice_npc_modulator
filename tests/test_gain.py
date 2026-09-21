# Copyright (C) 2026 Michael Cobb
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from votr.devices import DeviceSettings
from votr.engine import PassthroughEngine
from votr.gain import (
    MIC_GAIN_DEFAULT_DB,
    MIC_GAIN_MAX_DB,
    MIC_GAIN_MIN_DB,
    apply_mic_gain,
    clip_mic_gain_db,
    db_to_linear,
    db_to_slider,
    slider_to_db,
)
from votr.live import DuplexStream


def test_zero_db_is_unity() -> None:
    x = np.array([0.1, -0.25, 0.5, 0.0], dtype=np.float32)
    y = apply_mic_gain(x, 0.0)
    np.testing.assert_array_equal(y, x)
    assert db_to_linear(0.0) == 1.0
    assert MIC_GAIN_DEFAULT_DB == 0.0
    assert slider_to_db(db_to_slider(0.0)) == 0.0


def test_plus_six_db_is_about_two() -> None:
    x = np.full(8, 0.1, dtype=np.float32)
    y = apply_mic_gain(x, 6.0)
    np.testing.assert_allclose(y, x * db_to_linear(6.0), rtol=1e-6)
    np.testing.assert_allclose(db_to_linear(6.0), 10 ** (6.0 / 20.0))
    quieter = apply_mic_gain(x, -6.0)
    np.testing.assert_allclose(quieter, x * db_to_linear(-6.0), rtol=1e-6)


def test_gain_is_clipped_to_usable_range() -> None:
    assert clip_mic_gain_db(-99) == MIC_GAIN_MIN_DB
    assert clip_mic_gain_db(99) == MIC_GAIN_MAX_DB
    hot = apply_mic_gain(np.ones(4, dtype=np.float32), 99)
    np.testing.assert_allclose(hot, db_to_linear(MIC_GAIN_MAX_DB))


def test_mic_gain_persists_in_settings(tmp_path: Path) -> None:
    settings = DeviceSettings(mic_gain_db=6.0)
    settings.save(tmp_path)
    loaded = DeviceSettings.load(tmp_path)
    assert loaded.mic_gain_db == 6.0
    assert loaded.mic_gain_linear() == pytest.approx(db_to_linear(6.0))
    assert DeviceSettings.load(tmp_path / "missing").mic_gain_db == 0.0
    (tmp_path / "settings.json").write_text('{"mic_gain_db": 99}\n', encoding="utf-8")
    assert DeviceSettings.load(tmp_path).mic_gain_db == MIC_GAIN_MAX_DB
    (tmp_path / "settings.json").write_text("{}", encoding="utf-8")
    assert DeviceSettings.load(tmp_path).mic_gain_db == 0.0


def test_callback_zero_gain_does_not_change_samples() -> None:
    stream = DuplexStream(PassthroughEngine(block_size=256))
    assert stream.input_gain == 1.0
    dry = np.full((256, 1), 0.25, dtype=np.float32)
    out = np.zeros((256, 1), dtype=np.float32)
    stream.callback(dry, out, 256, None, None)
    np.testing.assert_array_equal(out[:, 0], dry[:, 0])
    assert dry[0, 0] == 0.25


def test_callback_applies_gain_before_engine() -> None:
    class Probe(PassthroughEngine):
        def process_block(self, block: np.ndarray) -> np.ndarray:
            self.seen = block.copy()
            return super().process_block(block)

    engine = Probe(block_size=256)
    stream = DuplexStream(engine)
    stream.input_gain = 2.0
    dry = np.full((256, 1), 0.2, dtype=np.float32)
    out = np.zeros((256, 1), dtype=np.float32)
    stream.callback(dry, out, 256, None, None)
    np.testing.assert_allclose(engine.seen, 0.4)
    np.testing.assert_allclose(out[:, 0], 0.4)
    assert dry[0, 0] == 0.2
    assert stream.input_peak == pytest.approx(0.4)

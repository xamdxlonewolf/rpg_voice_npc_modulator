# Copyright (C) 2026 Michael Cobb
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import numpy as np
import pytest

from votr.engine import PassthroughEngine
from votr.live import AudioDeviceError, DuplexStream, query_devices


def test_callback_writes_engine_output_not_dry() -> None:
    class SilenceEngine(PassthroughEngine):
        def process_block(self, block: np.ndarray) -> np.ndarray:
            return np.zeros_like(block)

    stream = DuplexStream(SilenceEngine(block_size=256, sample_rate=48_000))
    dry = np.full((256, 1), 0.25, dtype=np.float32)
    out = np.full((256, 1), 0.99, dtype=np.float32)
    stream.callback(dry, out, 256, None, None)
    np.testing.assert_array_equal(out, np.zeros_like(out))
    assert stream.callbacks == 1


def test_callback_silences_wrong_frame_count() -> None:
    engine = PassthroughEngine(block_size=256)
    stream = DuplexStream(engine)
    dry = np.ones((128, 1), dtype=np.float32)
    out = np.ones((128, 1), dtype=np.float32)
    stream.callback(dry, out, 128, None, None)
    np.testing.assert_array_equal(out, np.zeros_like(out))


def test_callback_uses_preallocated_input_buffer() -> None:
    engine = PassthroughEngine(block_size=256)
    stream = DuplexStream(engine)
    first = stream._in
    dry = np.ones((256, 1), dtype=np.float32)
    out = np.zeros((256, 1), dtype=np.float32)
    stream.callback(dry, out, 256, None, None)
    assert stream._in is first


def test_start_without_devices_raises() -> None:
    if query_devices():
        pytest.skip("audio devices present; cannot assert the empty-device path")
    stream = DuplexStream(PassthroughEngine())
    with pytest.raises(AudioDeviceError, match="no PortAudio devices"):
        stream.start()

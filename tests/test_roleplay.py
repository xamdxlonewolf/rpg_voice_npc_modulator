# Copyright (C) 2026 Michael Cobb
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from votr.devices import DeviceSettings
from votr.engine import PassthroughEngine
from votr.live import AudioDeviceError, BlockRing, DuplexStream, query_devices
from votr.roleplay import RoleplayError, RoleplayPath
from votr.session import Session
from votr.spikes.pitch_core import stretch_available


def test_silence_engine_never_writes_dry() -> None:
    class SilenceEngine(PassthroughEngine):
        def process_block(self, block: np.ndarray) -> np.ndarray:
            return np.zeros_like(block)

    stream = DuplexStream(SilenceEngine(block_size=256, sample_rate=48_000))
    dry = np.full((256, 1), 0.25, dtype=np.float32)
    out = np.full((256, 1), 0.99, dtype=np.float32)
    stream.callback(dry, out, 256, None, None)
    np.testing.assert_array_equal(out, np.zeros_like(out))


def test_passthrough_writes_engine_copy_not_indata() -> None:
    stream = DuplexStream(PassthroughEngine(block_size=256))
    dry = np.full((256, 1), 0.25, dtype=np.float32)
    out = np.zeros((256, 1), dtype=np.float32)
    stream.callback(dry, out, 256, None, None)
    np.testing.assert_array_equal(out[:, 0], dry[:, 0])
    out[0, 0] = 1.0
    assert dry[0, 0] == 0.25


def test_panic_mute_writes_silence() -> None:
    stream = DuplexStream(PassthroughEngine(block_size=256))
    stream.muted = True
    dry = np.full((256, 1), 0.5, dtype=np.float32)
    out = np.full((256, 1), 0.99, dtype=np.float32)
    stream.callback(dry, out, 256, None, None)
    np.testing.assert_array_equal(out, np.zeros_like(out))
    assert stream.output_peak == 0.0
    assert stream.input_peak > 0


def test_hold_to_talk_mutes_unless_held() -> None:
    stream = DuplexStream(PassthroughEngine(block_size=256))
    stream.hold_to_talk = True
    stream.talk_held = False
    dry = np.full((256, 1), 0.4, dtype=np.float32)
    out = np.ones((256, 1), dtype=np.float32)
    stream.callback(dry, out, 256, None, None)
    np.testing.assert_array_equal(out, np.zeros_like(out))
    stream.talk_held = True
    stream.callback(dry, out, 256, None, None)
    np.testing.assert_allclose(out[:, 0], dry[:, 0])


def test_monitor_ring_drops_oldest() -> None:
    ring = BlockRing(block_size=4, slots=2)
    first = np.array([1, 2, 3, 4], dtype=np.float32)
    second = np.array([5, 6, 7, 8], dtype=np.float32)
    third = np.array([9, 9, 9, 9], dtype=np.float32)
    ring.push(first)
    ring.push(second)
    ring.push(third)
    dest = np.zeros(4, dtype=np.float32)
    assert ring.pop(dest) is True
    np.testing.assert_array_equal(dest, second)
    ring.pop(dest)
    np.testing.assert_array_equal(dest, third)


def test_xrun_status_increments_counters() -> None:
    class Flags:
        input_overflow = True
        output_underflow = True

    stream = DuplexStream(PassthroughEngine(block_size=256))
    dry = np.zeros((256, 1), dtype=np.float32)
    out = np.zeros((256, 1), dtype=np.float32)
    stream.callback(dry, out, 256, None, Flags())
    assert stream.xruns == 1
    assert stream.overruns == 1
    assert stream.underruns == 1


def test_roleplay_cannot_start_without_cable(tmp_path: Path) -> None:
    if query_devices():
        pytest.skip("audio devices present; empty-device path not testable")
    path = RoleplayPath(PassthroughEngine(), DeviceSettings())
    ok, reason = path.can_start([])
    assert ok is False
    assert "Virtual Cable" in reason
    with pytest.raises(RoleplayError, match="Virtual Cable"):
        path.start()
    with pytest.raises(AudioDeviceError):
        DuplexStream(PassthroughEngine()).start()


@pytest.mark.skipif(not stretch_available(), reason="DSP Engine needs python-stretch")
def test_session_start_roleplay_stays_off_without_devices(tmp_path: Path) -> None:
    if query_devices():
        pytest.skip("audio devices present")
    session = Session(tmp_path)
    with pytest.raises(RoleplayError):
        session.start_roleplay()
    assert session.roleplay_on is False
    session.stop_roleplay()
    assert session.roleplay_on is False

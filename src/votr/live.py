# Copyright (C) 2026 Michael Cobb
# SPDX-License-Identifier: GPL-3.0-or-later

"""Duplex PortAudio stream: mic → Engine.process_block → output.

The callback uses preallocated buffers. It never copies the dry input to
``outdata``; only the Engine output is written. sounddevice/PortAudio
invokes the callback on its own I/O thread.
"""

from __future__ import annotations

import threading
from typing import Any

import numpy as np

from votr.engine import Engine


class AudioDeviceError(RuntimeError):
    """No usable audio device, or PortAudio failed to open a stream."""


def query_devices() -> list[dict[str, Any]]:
    """Return PortAudio devices, or an empty list if none / no host lib."""
    try:
        import sounddevice as sd
    except Exception:
        return []
    try:
        devices = sd.query_devices()
    except Exception:
        return []
    result = []
    for index, dev in enumerate(devices):
        item = dict(dev)
        item.setdefault("index", index)
        result.append(item)
    return result


class BlockRing:
    """Single-producer / single-consumer block ring for the Monitor."""

    def __init__(self, block_size: int, slots: int = 16) -> None:
        self.blocks = np.zeros((slots, block_size), dtype=np.float32)
        self.block_size = block_size
        self.slots = slots
        self._write = 0
        self._read = 0

    def push(self, block: np.ndarray) -> None:
        slot = self._write % self.slots
        count = min(block.size, self.block_size)
        self.blocks[slot, :count] = block[:count]
        if count < self.block_size:
            self.blocks[slot, count:] = 0
        self._write += 1
        unread = self._write - self._read
        if unread > self.slots:
            self._read = self._write - self.slots

    def pop(self, dest: np.ndarray) -> bool:
        if self._read >= self._write:
            dest.fill(0)
            return False
        dest[:] = self.blocks[self._read % self.slots]
        self._read += 1
        return True


class DuplexStream:
    """Full-duplex stream wrapping an Engine at a fixed block size."""

    def __init__(
        self,
        engine: Engine,
        *,
        input_device: int | str | None = None,
        output_device: int | str | None = None,
        latency: str = "low",
    ) -> None:
        self.engine = engine
        self.input_device = input_device
        self.output_device = output_device
        self.latency = latency
        self._in = np.zeros(engine.block_size, dtype=np.float32)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._started = threading.Event()
        self._error: BaseException | None = None
        self.xruns = 0
        self.underruns = 0
        self.overruns = 0
        self.callbacks = 0
        self.muted = False
        self.hold_to_talk = False
        self.talk_held = False
        self.input_peak = 0.0
        self.output_peak = 0.0
        self.lost = False
        self.monitor_ring: BlockRing | None = None
        # Linear capture gain; 1.0 is unity (no multiply in the callback).
        self.input_gain = 1.0

    def is_muted(self) -> bool:
        return self.muted or (self.hold_to_talk and not self.talk_held)

    @property
    def running(self) -> bool:
        return (
            self._thread is not None
            and self._thread.is_alive()
            and not self.lost
            and not self._stop.is_set()
        )

    def callback(
        self,
        indata: np.ndarray,
        outdata: np.ndarray,
        frames: int,
        time_info: object,
        status: object,
    ) -> None:
        """Allocation-light PortAudio callback. ``outdata`` is zeroed first."""
        outdata.fill(0)
        if status:
            self.xruns += 1
            if getattr(status, "input_overflow", False):
                self.overruns += 1
            if getattr(status, "output_underflow", False):
                self.underruns += 1
        if frames != self.engine.block_size:
            return
        self._in[:] = indata[:, 0]
        gain = self.input_gain
        if gain != 1.0:
            self._in *= gain
        self.input_peak = float(np.max(np.abs(self._in)))
        processed = self.engine.process_block(self._in)
        if self.is_muted():
            self.output_peak = 0.0
            self.callbacks += 1
            return
        count = min(processed.size, outdata.shape[0])
        outdata[:count, 0] = processed[:count]
        self.output_peak = float(np.max(np.abs(outdata[:count, 0])))
        if self.monitor_ring is not None:
            self.monitor_ring.push(processed[:count])
        self.callbacks += 1

    def start(self) -> None:
        if query_devices() == []:
            raise AudioDeviceError("no PortAudio devices available")
        self._stop.clear()
        self._started.clear()
        self._error = None
        self.lost = False
        self._thread = threading.Thread(
            target=self._run, name="votr-audio", daemon=True
        )
        self._thread.start()
        if not self._started.wait(timeout=5.0):
            self.stop()
            raise AudioDeviceError(str(self._error or "stream failed to start"))
        if self._error is not None:
            self.stop()
            raise AudioDeviceError(str(self._error))

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def _run(self) -> None:
        try:
            import sounddevice as sd

            stream = sd.Stream(
                samplerate=self.engine.sample_rate,
                blocksize=self.engine.block_size,
                dtype="float32",
                channels=1,
                callback=self.callback,
                latency=self.latency,
                device=(self.input_device, self.output_device),
            )
            stream.start()
            self._started.set()
            while not self._stop.wait(timeout=0.25):
                if not stream.active:
                    self.lost = True
                    break
            stream.stop()
            stream.close()
        except BaseException as exc:  # noqa: BLE001 — surface to start()
            self._error = exc
            self.lost = True
            self._started.set()

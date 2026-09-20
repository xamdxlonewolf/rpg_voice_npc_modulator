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
    return [dict(dev) for dev in devices]


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
        self.callbacks = 0

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
        if frames != self.engine.block_size:
            return
        self._in[:] = indata[:, 0]
        processed = self.engine.process_block(self._in)
        n = min(processed.size, outdata.shape[0])
        outdata[:n, 0] = processed[:n]
        self.callbacks += 1

    def start(self) -> None:
        if query_devices() == []:
            raise AudioDeviceError("no PortAudio devices available")
        self._stop.clear()
        self._started.clear()
        self._error = None
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
            self._stop.wait()
            stream.stop()
            stream.close()
        except BaseException as exc:  # noqa: BLE001 — surface to start()
            self._error = exc
            self._started.set()

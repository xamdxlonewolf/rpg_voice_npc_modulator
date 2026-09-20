# Copyright (C) 2026 Michael Cobb
# SPDX-License-Identifier: GPL-3.0-or-later

"""Roleplay Mode: mic → Engine → Virtual Cable. Dry Voice never leaves."""

from __future__ import annotations

import threading

from votr.devices import (
    DeviceSettings,
    device_ref,
    find_cable_input,
    find_mic,
    find_speakers,
    has_virtual_cable,
)
from votr.engine import Engine
from votr.live import AudioDeviceError, BlockRing, DuplexStream, query_devices


class RoleplayError(RuntimeError):
    """Roleplay Mode cannot start (missing cable, mic, or devices)."""


class RoleplayPath:
    """Live duplex path plus optional Monitor and panic mute."""

    def __init__(self, engine: Engine, settings: DeviceSettings) -> None:
        self.engine = engine
        self.settings = settings
        self.stream: DuplexStream | None = None
        self.monitor_error: str | None = None
        self._monitor_thread: threading.Thread | None = None
        self._monitor_stop = threading.Event()

    def can_start(self, devices: list | None = None) -> tuple[bool, str]:
        found = devices if devices is not None else query_devices()
        if not has_virtual_cable(found):
            return False, "A Virtual Cable is required. Open Discord setup."
        if find_mic(found, preferred=self.settings.mic_name) is None:
            return False, "No microphone is configured."
        return True, ""

    def start(self) -> None:
        devices = query_devices()
        ok, reason = self.can_start(devices)
        if not ok:
            raise RoleplayError(reason)
        cable = find_cable_input(devices, preferred=self.settings.cable_input_name)
        mic = find_mic(devices, preferred=self.settings.mic_name)
        if cable is None or mic is None:
            raise RoleplayError("A Virtual Cable is required. Open Discord setup.")
        self.stop()
        stream = DuplexStream(
            self.engine,
            input_device=device_ref(mic),
            output_device=device_ref(cable),
        )
        stream.hold_to_talk = self.settings.hold_to_talk
        if self.settings.monitor_on:
            stream.monitor_ring = BlockRing(self.engine.block_size)
        stream.start()
        self.stream = stream
        if self.settings.monitor_on:
            self._start_monitor(devices)

    def stop(self) -> None:
        self._stop_monitor()
        if self.stream is not None:
            self.stream.stop()
            self.stream = None

    def is_running(self) -> bool:
        return self.stream is not None and self.stream.running

    def toggle_panic(self) -> bool:
        if self.stream is None:
            return False
        self.stream.muted = not self.stream.muted
        return self.stream.muted

    def set_talk_held(self, held: bool) -> None:
        if self.stream is not None:
            self.stream.talk_held = held

    def set_hold_to_talk(self, enabled: bool) -> None:
        self.settings.hold_to_talk = enabled
        if self.stream is not None:
            self.stream.hold_to_talk = enabled

    def set_monitor(self, enabled: bool) -> None:
        self.settings.monitor_on = enabled
        if self.stream is None:
            return
        if enabled:
            if self.stream.monitor_ring is None:
                self.stream.monitor_ring = BlockRing(self.engine.block_size)
            self._start_monitor(query_devices())
        else:
            self._stop_monitor()
            self.stream.monitor_ring = None

    def recover(self) -> bool:
        if self.stream is None or not self.stream.lost:
            return False
        self.stop()
        try:
            self.start()
        except (RoleplayError, AudioDeviceError):
            return False
        return True

    def input_peak(self) -> float:
        return 0.0 if self.stream is None else self.stream.input_peak

    def output_peak(self) -> float:
        return 0.0 if self.stream is None else self.stream.output_peak

    def xrun_counts(self) -> tuple[int, int, int]:
        if self.stream is None:
            return 0, 0, 0
        return self.stream.underruns, self.stream.overruns, self.stream.xruns

    def is_muted(self) -> bool:
        return self.stream is not None and self.stream.is_muted()

    def _start_monitor(self, devices: list) -> None:
        self._stop_monitor()
        speakers = find_speakers(devices, preferred=self.settings.speaker_name)
        if speakers is None:
            self.monitor_error = "No speakers or headphones for Monitor."
            return
        ring = None if self.stream is None else self.stream.monitor_ring
        if ring is None:
            self.monitor_error = "Monitor buffer is not ready."
            return
        self.monitor_error = None
        self._monitor_stop.clear()
        engine = self.engine
        device = device_ref(speakers)

        def run() -> None:
            try:
                import sounddevice as sd

                def callback(outdata, frames, time_info, status) -> None:
                    outdata.fill(0)
                    if frames != engine.block_size:
                        return
                    ring.pop(outdata[:, 0])

                stream = sd.OutputStream(
                    samplerate=engine.sample_rate,
                    blocksize=engine.block_size,
                    dtype="float32",
                    channels=1,
                    callback=callback,
                    device=device,
                )
                stream.start()
                self._monitor_stop.wait()
                stream.stop()
                stream.close()
            except Exception as exc:  # noqa: BLE001 — surface in the panel
                self.monitor_error = str(exc)

        self._monitor_thread = threading.Thread(
            target=run, name="votr-monitor", daemon=True
        )
        self._monitor_thread.start()

    def _stop_monitor(self) -> None:
        self._monitor_stop.set()
        if self._monitor_thread is not None:
            self._monitor_thread.join(timeout=5.0)
            self._monitor_thread = None

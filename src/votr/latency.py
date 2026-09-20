# Copyright (C) 2026 Michael Cobb
# SPDX-License-Identifier: GPL-3.0-or-later

"""Latency Test: pick a block size. Never invent glass-to-glass numbers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from votr.devices import device_ref, find_cable_input, find_cable_output, find_mic
from votr.engine import Engine
from votr.live import DuplexStream, query_devices
from votr.spikes.latency import click_offset_samples

BLOCK_CANDIDATES = (256, 512, 1024)
TARGET_MS = 120.0
STRESS_SECONDS = 10.0


@dataclass(frozen=True)
class LatencyProbe:
    block_size: int
    method: str
    measured_ms: float | None
    engine_ms: float | None
    device_ms: float | None
    total_ms: float | None
    underruns: int | None
    blocked: str | None


def quality_for_block(block_size: int) -> int:
    if block_size <= 256:
        return 0
    if block_size <= 512:
        return 1
    return 2


def block_for_quality(quality: int) -> int:
    index = max(0, min(2, int(quality)))
    return BLOCK_CANDIDATES[index]


def engine_latency_ms(engine: Engine) -> float | None:
    frames = getattr(engine, "latency_frames", None)
    if not callable(frames):
        return 0.0
    return 1000.0 * float(frames()) / float(engine.sample_rate)


def device_reported_ms(*devices: dict[str, Any] | None) -> float | None:
    total = 0.0
    found = False
    keys = ("default_low_input_latency", "default_low_output_latency")
    for device in devices:
        if not device:
            continue
        for key in keys:
            if key in device:
                total += 1000.0 * float(device[key])
                found = True
    return total if found else None


def measure_click_loopback(
    engine: Engine,
    *,
    input_device: int | str,
    output_device: int | str,
) -> float | None:
    """Glass-to-glass click through the cable. None if it cannot be measured."""
    try:
        import sounddevice as sd
    except Exception:
        return None
    click_at = engine.block_size
    dry = np.zeros(engine.block_size * 8, dtype=np.float32)
    dry[click_at] = 0.9
    rendered = engine.render(dry)
    try:
        recorded = sd.playrec(
            rendered.reshape(-1, 1),
            samplerate=engine.sample_rate,
            channels=1,
            device=(input_device, output_device),
            dtype="float32",
        )
        sd.wait()
    except Exception:
        return None
    captured = np.asarray(recorded, dtype=np.float32).reshape(-1)
    if float(np.max(np.abs(captured))) < 0.01:
        return None
    offset = click_offset_samples(captured, click_at)
    if offset < 0:
        return None
    return 1000.0 * offset / float(engine.sample_rate)


def count_stream_xruns(
    engine: Engine,
    *,
    input_device: int | str | None,
    output_device: int | str | None,
    duration_s: float,
) -> int | None:
    if duration_s <= 0:
        return None
    if not query_devices():
        return None
    try:
        import time

        stream = DuplexStream(
            engine, input_device=input_device, output_device=output_device
        )
        stream.start()
        time.sleep(duration_s)
        xruns = stream.underruns + stream.xruns
        stream.stop()
        return int(xruns)
    except Exception:
        return None


def probe_block(
    engine: Engine,
    *,
    devices: list[dict[str, Any]],
    duration_s: float,
) -> LatencyProbe:
    engine_ms = engine_latency_ms(engine)
    mic = find_mic(devices)
    cable_in = find_cable_input(devices)
    cable_out = find_cable_output(devices)
    device_ms = device_reported_ms(mic, cable_in, cable_out)
    blocked = None
    if not devices:
        blocked = "no PortAudio devices (cannot measure glass-to-glass)"
    elif cable_in is None or cable_out is None:
        blocked = "Virtual Cable not capturable; engine + device-reported only"

    measured_ms = None
    method = "blocked"
    if cable_in is not None and cable_out is not None:
        measured_ms = measure_click_loopback(
            engine,
            input_device=device_ref(cable_out),
            output_device=device_ref(cable_in),
        )
        if measured_ms is not None:
            method = "glass_to_glass"
            blocked = None
    if method != "glass_to_glass":
        if engine_ms is not None or device_ms is not None:
            method = "engine_plus_device" if device_ms is not None else "engine_only"

    if measured_ms is not None:
        total_ms = measured_ms
    elif engine_ms is not None and device_ms is not None:
        total_ms = engine_ms + device_ms
    else:
        total_ms = engine_ms

    underruns = None
    if devices and duration_s > 0:
        underruns = count_stream_xruns(
            engine,
            input_device=device_ref(mic) if mic else None,
            output_device=device_ref(cable_in) if cable_in else None,
            duration_s=duration_s,
        )
    return LatencyProbe(
        block_size=engine.block_size,
        method=method,
        measured_ms=measured_ms,
        engine_ms=engine_ms,
        device_ms=device_ms,
        total_ms=total_ms,
        underruns=underruns,
        blocked=blocked,
    )


def run_latency_test(
    make_engine: Callable[[int], Engine],
    *,
    devices: list[dict[str, Any]] | None = None,
    duration_s: float = STRESS_SECONDS,
) -> list[LatencyProbe]:
    found = devices if devices is not None else query_devices()
    return [
        probe_block(make_engine(block), devices=found, duration_s=duration_s)
        for block in BLOCK_CANDIDATES
    ]


def choose_probe(probes: list[LatencyProbe]) -> LatencyProbe:
    if not probes:
        raise ValueError("no latency probes")
    measured = [row for row in probes if row.total_ms is not None]
    clean = [row for row in measured if row.underruns == 0]
    under = [
        row
        for row in clean
        if row.total_ms is not None and row.total_ms <= TARGET_MS
    ]
    if under:
        return min(under, key=lambda row: row.block_size)
    if clean:
        return min(clean, key=lambda row: row.block_size)
    if measured:
        return min(measured, key=lambda row: row.block_size)
    return probes[0]


def format_latency_report(probe: LatencyProbe) -> str:
    if probe.method == "glass_to_glass" and probe.measured_ms is not None:
        return (
            f"Glass-to-glass {probe.measured_ms:.0f} ms "
            f"(block {probe.block_size})."
        )
    parts = []
    if probe.engine_ms is not None:
        parts.append(f"Engine {probe.engine_ms:.0f} ms")
    if probe.device_ms is not None:
        parts.append(f"device-reported {probe.device_ms:.0f} ms")
    body = " + ".join(parts) if parts else "No latency number"
    extra = f" {probe.blocked}" if probe.blocked else ""
    return f"{body}.{extra} Not glass-to-glass."

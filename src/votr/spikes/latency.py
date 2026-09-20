# Copyright (C) 2026 Michael Cobb
# SPDX-License-Identifier: GPL-3.0-or-later

"""S0.4 loopback harness. Glass-to-glass needs VB-CABLE on Windows."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

CABLE_OUTPUT_PATTERNS = ("cable output", "vb-audio virtual cable")
CABLE_INPUT_PATTERNS = ("cable input", "vb-audio virtual cable")


def find_named_device(
    devices: list[dict[str, Any]], patterns: tuple[str, ...], *, kind: str
) -> dict[str, Any] | None:
    """Match a PortAudio device by name substring (case-insensitive)."""
    key = "max_input_channels" if kind == "input" else "max_output_channels"
    for device in devices:
        name = str(device.get("name", "")).lower()
        if any(part in name for part in patterns) and int(device.get(key, 0)) > 0:
            return device
    return None


def click_offset_samples(captured: np.ndarray, click_at: int) -> int:
    """Return peak index minus the known click index."""
    peak = int(np.argmax(np.abs(np.asarray(captured, dtype=np.float32))))
    return peak - click_at


@dataclass(frozen=True)
class LatencyAttempt:
    host_api: str
    block_size: int
    exclusive: bool
    measured_ms: float | None
    blocked: str | None


def collect_attempts() -> list[LatencyAttempt]:
    """Record what this machine can and cannot measure."""
    from votr.live import query_devices

    devices = query_devices()
    cable_out = find_named_device(devices, CABLE_OUTPUT_PATTERNS, kind="input")
    cable_in = find_named_device(devices, CABLE_INPUT_PATTERNS, kind="output")
    blocked = None
    if not devices:
        blocked = "no PortAudio devices (Linux cloud VM has empty ALSA/OSS)"
    elif cable_out is None or cable_in is None:
        blocked = "VB-CABLE devices not present; needs Michael's Windows machine"

    rows = []
    for host_api, exclusive in (("WASAPI", False), ("WASAPI", True)):
        for hop in (256, 512):
            rows.append(
                LatencyAttempt(
                    host_api=host_api,
                    block_size=hop,
                    exclusive=exclusive,
                    measured_ms=None,
                    blocked=blocked,
                )
            )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    for row in collect_attempts():
        print(asdict(row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

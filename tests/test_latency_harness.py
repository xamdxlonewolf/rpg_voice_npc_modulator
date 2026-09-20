# Copyright (C) 2026 Michael Cobb
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import numpy as np

from votr.spikes.latency import (
    CABLE_OUTPUT_PATTERNS,
    click_offset_samples,
    collect_attempts,
    find_named_device,
)


def test_find_cable_output_by_name() -> None:
    devices = [
        {"name": "Speakers", "max_input_channels": 0, "max_output_channels": 2},
        {
            "name": "CABLE Output (VB-Audio Virtual Cable)",
            "max_input_channels": 2,
            "max_output_channels": 0,
        },
    ]
    found = find_named_device(devices, CABLE_OUTPUT_PATTERNS, kind="input")
    assert found is not None
    assert "CABLE Output" in found["name"]


def test_click_offset_samples() -> None:
    captured = np.zeros(1000, dtype=np.float32)
    captured[480] = 0.8
    assert click_offset_samples(captured, click_at=400) == 80


def test_collect_attempts_are_blocked_without_cable() -> None:
    rows = collect_attempts()
    assert len(rows) == 4
    assert all(row.measured_ms is None for row in rows)
    assert all(row.blocked for row in rows)

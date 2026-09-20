# Copyright (C) 2026 Michael Cobb
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

from pathlib import Path

import pytest

from votr.engine import PassthroughEngine
from votr.latency import (
    LatencyProbe,
    choose_probe,
    format_latency_report,
    quality_for_block,
    run_latency_test,
)
from votr.session import Session
from votr.spikes.pitch_core import stretch_available


def test_choose_smallest_under_target_with_zero_underruns() -> None:
    probes = [
        LatencyProbe(256, "glass_to_glass", 90.0, 20.0, 10.0, 90.0, 1, None),
        LatencyProbe(512, "glass_to_glass", 110.0, 20.0, 10.0, 110.0, 0, None),
        LatencyProbe(1024, "glass_to_glass", 80.0, 20.0, 10.0, 80.0, 0, None),
    ]
    chosen = choose_probe(probes)
    assert chosen.block_size == 512


def test_choose_falls_back_to_smallest_estimate() -> None:
    probes = [
        LatencyProbe(256, "engine_only", None, 40.0, None, 40.0, None, "no devices"),
        LatencyProbe(512, "engine_only", None, 40.0, None, 40.0, None, "no devices"),
        LatencyProbe(1024, "engine_only", None, 40.0, None, 40.0, None, "no devices"),
    ]
    assert choose_probe(probes).block_size == 256


def test_choose_blocked_does_not_invent_ms() -> None:
    probes = [
        LatencyProbe(256, "blocked", None, None, None, None, None, "no devices"),
        LatencyProbe(512, "blocked", None, None, None, None, None, "no devices"),
    ]
    chosen = choose_probe(probes)
    report = format_latency_report(chosen).lower()
    assert chosen.measured_ms is None
    assert chosen.total_ms is None
    assert "not glass-to-glass" in report


def test_quality_mapping() -> None:
    assert quality_for_block(256) == 0
    assert quality_for_block(512) == 1
    assert quality_for_block(1024) == 2


def test_run_latency_test_without_devices_is_honest() -> None:
    from votr.live import query_devices

    if query_devices():
        pytest.skip("audio devices present")

    def make(block: int) -> PassthroughEngine:
        return PassthroughEngine(block_size=block)

    probes = run_latency_test(make, devices=[], duration_s=0.0)
    assert len(probes) == 3
    assert all(row.measured_ms is None for row in probes)
    assert all(row.method != "glass_to_glass" for row in probes)
    report = format_latency_report(probes[0])
    assert "Not glass-to-glass" in report


@pytest.mark.skipif(not stretch_available(), reason="python-stretch required")
def test_session_persists_latency_without_fake_glass(tmp_path: Path) -> None:
    from votr.live import query_devices

    if query_devices():
        pytest.skip("audio devices present")
    session = Session(tmp_path)
    probe = session.run_latency_test(duration_s=0.0)
    assert probe.measured_ms is None
    loaded = session.settings
    assert loaded.latency_ms is None
    assert loaded.latency_method != "glass_to_glass"
    assert loaded.latency_blocked
    again = type(session.settings).load(tmp_path)
    assert again.latency_method == loaded.latency_method
    assert again.latency_ms is None

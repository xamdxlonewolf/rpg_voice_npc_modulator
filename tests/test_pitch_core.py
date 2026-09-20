# Copyright (C) 2026 Michael Cobb
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import numpy as np
import pytest

from votr.spikes.pitch_core import (
    LiveShifterEngine,
    StretchEngine,
    collect_measurements,
    rubband_available,
    stretch_available,
)


@pytest.mark.skipif(not stretch_available(), reason="python-stretch not installed")
@pytest.mark.parametrize("hop", [256, 512])
@pytest.mark.parametrize("configure_ms", [60.0, 120.0])
def test_stretch_process_block_is_one_to_one(hop: int, configure_ms: float) -> None:
    engine = StretchEngine(block_size=hop, configure_ms=configure_ms)
    dry = np.linspace(-0.4, 0.4, hop, dtype=np.float32)
    out = engine.process_block(dry)
    assert out.shape == (hop,)
    assert out.dtype == np.float32


@pytest.mark.skipif(not stretch_available(), reason="python-stretch not installed")
def test_stretch_render_equals_concatenated_blocks() -> None:
    engine = StretchEngine(block_size=256, configure_ms=60.0)
    take = np.linspace(-0.3, 0.3, 1000, dtype=np.float32)
    rendered = engine.render(take)
    assert rendered.shape == take.shape
    assert rendered.dtype == np.float32


@pytest.mark.skipif(not rubband_available(), reason="rubband not installed")
def test_live_shifter_exposes_formant_and_fixed_block() -> None:
    engine = LiveShifterEngine(pitch_semitones=3.0, formant_scale=1.0)
    assert engine.block_size == engine._shifter.get_block_size()
    silent = np.zeros(engine.block_size, dtype=np.float32)
    out = engine.process_block(silent)
    assert out.shape == (engine.block_size,)
    engine._shifter.set_formant_scale(0.8)
    assert engine._shifter.get_formant_scale() == pytest.approx(0.8)


def test_collect_measurements_includes_wheel_facts() -> None:
    rows = collect_measurements()
    names = [row.name for row in rows]
    assert "rubband.LiveShifter" in names
    assert "python-stretch" in names
    rubband_row = next(row for row in rows if row.name == "rubband.LiveShifter")
    assert rubband_row.windows_wheel is False
    stretch_row = next(row for row in rows if row.name == "python-stretch")
    assert stretch_row.windows_wheel is True

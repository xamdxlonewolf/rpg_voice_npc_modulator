# Copyright (C) 2026 Michael Cobb
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import numpy as np
import pytest

from votr.dsp import DspEngine
from votr.spikes.pitch_core import stretch_available


@pytest.mark.skipif(not stretch_available(), reason="python-stretch required")
def test_crossfade_stays_finite() -> None:
    engine = DspEngine(prefer_rubband=False)
    t = np.arange(engine.block_size, dtype=np.float32) / engine.sample_rate
    block = (0.3 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
    engine.process_block(block)
    engine.set_params({"pitch_semitones": -6.0}, crossfade_ms=30.0)
    assert engine._fade_left > 0
    out = engine.process_block(block)
    assert np.isfinite(out).all()
    assert float(np.max(np.abs(out))) <= 1.05

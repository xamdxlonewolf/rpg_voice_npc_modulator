# Copyright (C) 2026 Michael Cobb
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import numpy as np
import pytest

from votr.dsp import DspEngine
from votr.macros import load_macros
from votr.spikes.pitch_core import stretch_available

pytestmark = pytest.mark.skipif(
    not stretch_available(), reason="python-stretch required for DSP Engine tests"
)


def test_silence_in_silence_out() -> None:
    engine = DspEngine(prefer_rubband=False)
    silent = np.zeros(engine.block_size, dtype=np.float32)
    # Flush shifter memory, then a silent block must stay near silent.
    for _ in range(8):
        engine.process_block(silent)
    out = engine.process_block(silent)
    assert float(np.max(np.abs(out))) < 0.02


def test_rms_is_bounded() -> None:
    engine = DspEngine(prefer_rubband=False)
    engine.set_params({"growl": 1.0, "room": 0.5, "hollow": 0.4})
    t = np.arange(engine.block_size, dtype=np.float32) / engine.sample_rate
    loud = (0.95 * np.sin(2 * np.pi * 180 * t)).astype(np.float32)
    out = engine.process_block(loud)
    assert float(np.sqrt(np.mean(out * out))) < 1.0
    assert float(np.max(np.abs(out))) <= 1.0 + 1e-3


def test_latency_frames_reported() -> None:
    engine = DspEngine(prefer_rubband=False)
    assert engine.latency_frames() > 0


def test_render_uses_process_block() -> None:
    engine = DspEngine(prefer_rubband=False)
    take = np.linspace(-0.2, 0.2, engine.block_size * 3 + 40, dtype=np.float32)
    rendered = engine.render(take)
    assert rendered.shape == take.shape
    assert rendered.dtype == np.float32


def test_apply_macro_patches_only_defined_keys() -> None:
    engine = DspEngine(macros=load_macros(), prefer_rubband=False)
    engine.set_params({"room": 0.8, "growl": 0.0})
    engine.apply_macro("gravelly")
    params = engine.params()
    assert params["growl"] == pytest.approx(0.55)
    assert "room" not in load_macros()["gravelly"].params
    assert params["room"] == pytest.approx(0.8)

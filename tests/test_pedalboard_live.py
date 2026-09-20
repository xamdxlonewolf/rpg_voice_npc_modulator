# Copyright (C) 2026 Michael Cobb
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import numpy as np
import pytest

from votr.spikes.pedalboard_live import (
    PedalboardEffectsEngine,
    _sine,
    boundary_step_ratio,
    pedalboard_available,
)

pytestmark = pytest.mark.skipif(
    not pedalboard_available(), reason="pedalboard not installed"
)


def test_reset_false_matches_offline_and_is_smooth() -> None:
    import pedalboard as pb

    hop = 256
    sine = _sine(1.0, 48_000)
    plugin = pb.LowpassFilter(cutoff_frequency_hz=1200)
    offline = pb.LowpassFilter(cutoff_frequency_hz=1200).process(sine, 48_000)
    chunks = [
        plugin.process(sine[i : i + hop], 48_000, reset=False)
        for i in range(0, sine.size, hop)
    ]
    streamed = np.concatenate(chunks)[: sine.size]
    np.testing.assert_allclose(streamed, offline, atol=1e-6)
    assert boundary_step_ratio(streamed, hop) < 1.2


def test_reset_true_clicks_on_lowpass_boundaries() -> None:
    import pedalboard as pb

    hop = 256
    sine = _sine(1.0, 48_000)
    plugin = pb.LowpassFilter(cutoff_frequency_hz=1200)
    chunks = [
        plugin.process(sine[i : i + hop], 48_000, reset=True)
        for i in range(0, sine.size, hop)
    ]
    streamed = np.concatenate(chunks)[: sine.size]
    assert boundary_step_ratio(streamed, hop) > 10


def test_effects_engine_process_block_length() -> None:
    engine = PedalboardEffectsEngine(block_size=256, reset=False)
    dry = np.linspace(-0.2, 0.2, 256, dtype=np.float32)
    out = engine.process_block(dry)
    assert out.shape[0] == 256
    assert np.isfinite(out).all()

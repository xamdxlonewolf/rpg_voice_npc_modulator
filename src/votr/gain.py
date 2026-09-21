# Copyright (C) 2026 Michael Cobb
# SPDX-License-Identifier: GPL-3.0-or-later

"""Microphone capture gain, applied before Engine / Preview / Roleplay.

0 dB is unity: the samples are returned unchanged, with no hidden extra boost.
"""

from __future__ import annotations

import numpy as np

MIC_GAIN_MIN_DB = -12.0
MIC_GAIN_MAX_DB = 24.0
MIC_GAIN_DEFAULT_DB = 0.0
# 0.1 dB ticks so the slider can sit exactly on 0 dB.
MIC_GAIN_STEP_DB = 0.1
CLIPPING_PEAK = 0.99


def clip_mic_gain_db(db: float) -> float:
    value = float(db)
    if value < MIC_GAIN_MIN_DB:
        return MIC_GAIN_MIN_DB
    if value > MIC_GAIN_MAX_DB:
        return MIC_GAIN_MAX_DB
    return value


def db_to_linear(db: float) -> float:
    """Linear amplitude for a dB value. 0 dB → 1.0 exactly."""
    if float(db) == 0.0:
        return 1.0
    return float(10.0 ** (clip_mic_gain_db(db) / 20.0))


def gain_slider_steps() -> int:
    span = MIC_GAIN_MAX_DB - MIC_GAIN_MIN_DB
    return int(round(span / MIC_GAIN_STEP_DB))


def db_to_slider(db: float) -> int:
    clipped = clip_mic_gain_db(db)
    return int(round((clipped - MIC_GAIN_MIN_DB) / MIC_GAIN_STEP_DB))


def slider_to_db(pos: int) -> float:
    return clip_mic_gain_db(MIC_GAIN_MIN_DB + int(pos) * MIC_GAIN_STEP_DB)


def format_mic_gain(db: float) -> str:
    clipped = clip_mic_gain_db(db)
    label = f"{clipped:+.1f} dB"
    if clipped == 0.0:
        return f"{label} (unity — no boost)"
    return label


def apply_mic_gain(samples: np.ndarray, gain_db: float) -> np.ndarray:
    """Scale captured mic audio. 0 dB leaves the sample values unchanged."""
    audio = np.asarray(samples, dtype=np.float32)
    db = clip_mic_gain_db(gain_db)
    if db == 0.0:
        return np.array(audio, dtype=np.float32, copy=True)
    return (audio * db_to_linear(db)).astype(np.float32)


def peak_meter(samples: np.ndarray) -> tuple[float, bool]:
    """(peak 0..1, clipping). Empty audio is silence, not clipping."""
    audio = np.asarray(samples, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        return 0.0, False
    peak = float(np.max(np.abs(audio)))
    return peak, peak >= CLIPPING_PEAK

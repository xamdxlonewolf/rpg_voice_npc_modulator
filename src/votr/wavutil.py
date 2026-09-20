# Copyright (C) 2026 Michael Cobb
# SPDX-License-Identifier: GPL-3.0-or-later

"""Mono float32 WAV helpers for Takes."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np


def write_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    scaled = np.asarray(samples, dtype=np.float32) * 32767.0
    pcm = np.clip(np.rint(scaled), -32768, 32767)
    pcm = pcm.astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wav:
        rate = wav.getframerate()
        raw = wav.readframes(wav.getnframes())
        width = wav.getsampwidth()
        channels = wav.getnchannels()
    if width == 2:
        pcm = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32767.0
    else:
        pcm = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32767.0
    if channels > 1:
        pcm = pcm.reshape(-1, channels).mean(axis=1)
    return pcm.astype(np.float32), rate

# Copyright (C) 2026 Michael Cobb
# SPDX-License-Identifier: GPL-3.0-or-later

"""Delayed Preview: record a Take, render through the Engine, play speakers."""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path
from typing import Any

import numpy as np

from votr.dsp import DspEngine
from votr.engine import DEFAULT_SAMPLE_RATE
from votr.live import query_devices
from votr.wavutil import read_wav, write_wav

MAX_TAKE_SECONDS = 15
SAMPLE_RATE = DEFAULT_SAMPLE_RATE


def load_sample_take() -> np.ndarray:
    path = files("votr.assets").joinpath("sample_take.wav")
    samples, rate = read_wav(Path(str(path)))
    if rate != SAMPLE_RATE:
        # Bundled file is written at 48 kHz; keep a safe fallback.
        return samples
    return samples


def takes_dir(data_dir: Path) -> Path:
    path = Path(data_dir) / "takes"
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_last_take(data_dir: Path, samples: np.ndarray) -> Path:
    path = takes_dir(data_dir) / "last.wav"
    write_wav(path, samples, SAMPLE_RATE)
    return path


def keep_take(data_dir: Path, samples: np.ndarray, name: str) -> Path:
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name.strip())
    path = takes_dir(data_dir) / f"{safe or 'take'}.wav"
    write_wav(path, samples, SAMPLE_RATE)
    return path


def input_devices(devices: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    found = devices if devices is not None else query_devices()
    return [dev for dev in found if int(dev.get("max_input_channels", 0)) > 0]


def speaker_devices(
    devices: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    found = devices if devices is not None else query_devices()
    speakers = []
    for dev in found:
        name = str(dev.get("name", "")).lower()
        if int(dev.get("max_output_channels", 0)) <= 0:
            continue
        if "cable input" in name:
            continue
        speakers.append(dev)
    return speakers


def render_take(engine: DspEngine, take: np.ndarray) -> np.ndarray:
    return engine.render(np.asarray(take, dtype=np.float32))


def play_on_speakers(
    samples: np.ndarray,
    *,
    sample_rate: int = SAMPLE_RATE,
    device: int | str | None = None,
) -> bool:
    """Play to speakers. Returns False when no output device is available."""
    if not speaker_devices() and device is None:
        return False
    try:
        import sounddevice as sd

        sd.stop()
        sd.play(np.asarray(samples, dtype=np.float32), sample_rate, device=device)
        return True
    except Exception:
        return False


def stop_playback() -> None:
    try:
        import sounddevice as sd

        sd.stop()
    except Exception:
        return

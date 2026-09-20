# Copyright (C) 2026 Michael Cobb
# SPDX-License-Identifier: GPL-3.0-or-later

"""Virtual Cable detection, device picks, and Discord-wizard helpers.

Virtual Cable drivers are detected and linked (ADR-0006), never bundled.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from votr.engine import DEFAULT_SAMPLE_RATE
from votr.live import query_devices
from votr.preview import input_devices, speaker_devices

VB_CABLE_URL = "https://vb-audio.com/Cable/"
BLACKHOLE_URL = "https://existential.audio/blackhole/"

# Output device the app writes the Character Voice into.
CABLE_INPUT_PATTERNS = (
    "cable input",
    "blackhole",
    "votr_cable",
    "voice of the realm",
)
# Input device Discord should use (reads the cable).
CABLE_OUTPUT_PATTERNS = (
    "cable output",
    "blackhole",
    "votr_cable",
    "voice of the realm",
)

PACTL_SINK_NAME = "votr_cable"
PACTL_SINK_DESC = "VoiceOfTheRealm"


def _name(device: dict[str, Any]) -> str:
    return str(device.get("name", ""))


def _fold(text: str) -> str:
    return "".join(ch for ch in text.lower() if ch.isalnum())


def _channels(device: dict[str, Any], key: str) -> int:
    return int(device.get(key, 0))


def device_ref(device: dict[str, Any]) -> int | str:
    if "index" in device:
        return int(device["index"])
    return _name(device)


def _preferred_or_first(
    matches: list[dict[str, Any]], preferred: str
) -> dict[str, Any] | None:
    if preferred:
        for device in matches:
            if _name(device) == preferred:
                return device
    return matches[0] if matches else None


def find_named_outputs(
    devices: list[dict[str, Any]], patterns: tuple[str, ...]
) -> list[dict[str, Any]]:
    found = []
    for device in devices:
        name = _name(device).lower()
        if _channels(device, "max_output_channels") <= 0:
            continue
        if "cable output" in name:
            continue
        folded = _fold(name)
        if any(_fold(part) in folded for part in patterns):
            found.append(device)
    return found


def find_named_inputs(
    devices: list[dict[str, Any]], patterns: tuple[str, ...]
) -> list[dict[str, Any]]:
    found = []
    for device in devices:
        name = _name(device).lower()
        if _channels(device, "max_input_channels") <= 0:
            continue
        if "cable input" in name:
            continue
        folded = _fold(name)
        if any(_fold(part) in folded for part in patterns):
            found.append(device)
    return found


def find_cable_input(
    devices: list[dict[str, Any]] | None = None, *, preferred: str = ""
) -> dict[str, Any] | None:
    found = devices if devices is not None else query_devices()
    return _preferred_or_first(
        find_named_outputs(found, CABLE_INPUT_PATTERNS), preferred
    )


def find_cable_output(
    devices: list[dict[str, Any]] | None = None, *, preferred: str = ""
) -> dict[str, Any] | None:
    found = devices if devices is not None else query_devices()
    return _preferred_or_first(
        find_named_inputs(found, CABLE_OUTPUT_PATTERNS), preferred
    )


def has_virtual_cable(devices: list[dict[str, Any]] | None = None) -> bool:
    return find_cable_input(devices) is not None


def mic_devices(devices: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    found = devices if devices is not None else query_devices()
    skip = (
        "cable output",
        "cable input",
        "votr_cable",
        "blackhole",
        "voice of the realm",
    )
    mics = []
    for device in input_devices(found):
        folded = _fold(_name(device))
        if any(_fold(part) in folded for part in skip):
            continue
        mics.append(device)
    return mics


def find_mic(
    devices: list[dict[str, Any]] | None = None, *, preferred: str = ""
) -> dict[str, Any] | None:
    found = devices if devices is not None else query_devices()
    return _preferred_or_first(mic_devices(found), preferred)


def find_speakers(
    devices: list[dict[str, Any]] | None = None, *, preferred: str = ""
) -> dict[str, Any] | None:
    found = devices if devices is not None else query_devices()
    return _preferred_or_first(speaker_devices(found), preferred)


def monitor_looks_like_speakers(name: str) -> bool:
    lowered = name.lower()
    if any(
        word in lowered for word in ("headphone", "headset", "earphone", "airpod")
    ):
        return False
    return True


def linux_null_sink_command() -> list[str]:
    return [
        "pactl",
        "load-module",
        "module-null-sink",
        f"sink_name={PACTL_SINK_NAME}",
        f"sink_properties=device.description={PACTL_SINK_DESC}",
    ]


def create_linux_null_sink() -> tuple[bool, str]:
    try:
        result = subprocess.run(
            linux_null_sink_command(),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except FileNotFoundError:
        return False, "pactl is not installed."
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "pactl failed").strip()
        return False, detail
    return True, (result.stdout or "Null sink created.").strip()


def make_test_tone(
    *, seconds: float = 1.0, freq: float = 440.0, sample_rate: int = DEFAULT_SAMPLE_RATE
) -> np.ndarray:
    count = max(1, int(sample_rate * seconds))
    time = np.arange(count, dtype=np.float32) / float(sample_rate)
    return (0.2 * np.sin(2.0 * np.pi * freq * time)).astype(np.float32)


def play_to_cable(
    samples: np.ndarray,
    *,
    device: int | str | None,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> bool:
    if device is None:
        return False
    try:
        import sounddevice as sd

        sd.play(np.asarray(samples, dtype=np.float32), sample_rate, device=device)
        return True
    except Exception:
        return False


@dataclass
class DeviceSettings:
    mic_name: str = ""
    speaker_name: str = ""
    cable_input_name: str = ""
    hold_to_talk: bool = False
    monitor_on: bool = False
    wizard_completed: bool = False

    def save(self, data_dir: Path) -> Path:
        path = Path(data_dir) / "settings.json"
        path.write_text(
            json.dumps(asdict(self), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return path

    @classmethod
    def load(cls, data_dir: Path) -> DeviceSettings:
        path = Path(data_dir) / "settings.json"
        if not path.exists():
            return cls()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                return cls()
            return cls(
                mic_name=str(data.get("mic_name", "")),
                speaker_name=str(data.get("speaker_name", "")),
                cable_input_name=str(data.get("cable_input_name", "")),
                hold_to_talk=bool(data.get("hold_to_talk", False)),
                monitor_on=bool(data.get("monitor_on", False)),
                wizard_completed=bool(data.get("wizard_completed", False)),
            )
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            return cls()

# Copyright (C) 2026 Michael Cobb
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

from pathlib import Path

import numpy as np

from votr.devices import (
    BLACKHOLE_URL,
    CABLE_INPUT_PATTERNS,
    VB_CABLE_URL,
    DeviceSettings,
    create_linux_null_sink,
    find_cable_input,
    find_cable_output,
    find_mic,
    has_virtual_cable,
    linux_null_sink_command,
    make_test_tone,
    mic_devices,
    monitor_looks_like_speakers,
    play_to_cable,
)

DEVICES = [
    {
        "name": "Microphone",
        "max_input_channels": 1,
        "max_output_channels": 0,
        "index": 0,
    },
    {
        "name": "Speakers",
        "max_input_channels": 0,
        "max_output_channels": 2,
        "index": 1,
    },
    {
        "name": "CABLE Input (VB-Audio Virtual Cable)",
        "max_input_channels": 0,
        "max_output_channels": 2,
        "index": 2,
    },
    {
        "name": "CABLE Output (VB-Audio Virtual Cable)",
        "max_input_channels": 2,
        "max_output_channels": 0,
        "index": 3,
    },
]


def test_find_vb_cable_by_name() -> None:
    cable_in = find_cable_input(DEVICES)
    cable_out = find_cable_output(DEVICES)
    assert cable_in is not None
    assert "CABLE Input" in cable_in["name"]
    assert cable_out is not None
    assert "CABLE Output" in cable_out["name"]
    assert has_virtual_cable(DEVICES)
    assert not has_virtual_cable(DEVICES[:2])


def test_mic_devices_skip_cable() -> None:
    assert [dev["name"] for dev in mic_devices(DEVICES)] == ["Microphone"]
    assert find_mic(DEVICES, preferred="Microphone")["name"] == "Microphone"


def test_linux_null_sink_is_detectable() -> None:
    linux = [
        {
            "name": "VoiceOfTheRealm",
            "max_input_channels": 0,
            "max_output_channels": 2,
            "index": 0,
        },
        {
            "name": "VoiceOfTheRealm.monitor",
            "max_input_channels": 2,
            "max_output_channels": 0,
            "index": 1,
        },
    ]
    assert find_cable_input(linux)["name"] == "VoiceOfTheRealm"
    assert find_cable_output(linux)["name"] == "VoiceOfTheRealm.monitor"


def test_pactl_command_and_missing_binary(monkeypatch) -> None:
    command = linux_null_sink_command()
    assert command[:3] == ["pactl", "load-module", "module-null-sink"]
    assert "votr_cable" in " ".join(command)
    assert "VoiceOfTheRealm" in " ".join(command)

    def boom(*_args, **_kwargs):
        raise FileNotFoundError("pactl")

    monkeypatch.setattr("votr.devices.subprocess.run", boom)
    ok, detail = create_linux_null_sink()
    assert ok is False
    assert "pactl" in detail


def test_create_linux_null_sink_success(monkeypatch) -> None:
    class Result:
        returncode = 0
        stdout = "42"
        stderr = ""

    monkeypatch.setattr(
        "votr.devices.subprocess.run", lambda *_args, **_kwargs: Result()
    )
    ok, detail = create_linux_null_sink()
    assert ok is True
    assert "42" in detail


def test_settings_round_trip(tmp_path: Path) -> None:
    settings = DeviceSettings(mic_name="Microphone", wizard_completed=True)
    settings.save(tmp_path)
    loaded = DeviceSettings.load(tmp_path)
    assert loaded.mic_name == "Microphone"
    assert loaded.wizard_completed is True
    assert loaded.mic_gain_db == 0.0
    settings = DeviceSettings(mic_name="Microphone", mic_gain_db=12.0)
    settings.save(tmp_path)
    assert DeviceSettings.load(tmp_path).mic_gain_db == 12.0
    assert DeviceSettings.load(tmp_path / "missing").mic_name == ""


def test_test_tone_and_play_without_device() -> None:
    tone = make_test_tone(seconds=0.05)
    assert tone.dtype == np.float32
    assert tone.size > 10
    assert play_to_cable(tone, device=None) is False


def test_monitor_warning_for_speakers() -> None:
    assert monitor_looks_like_speakers("Speakers (Realtek)") is True
    assert monitor_looks_like_speakers("Headphones") is False
    assert "vb-audio.com" in VB_CABLE_URL
    assert "blackhole" in BLACKHOLE_URL.lower()
    assert "cable input" in CABLE_INPUT_PATTERNS

# Copyright (C) 2026 Michael Cobb
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from votr.loopback import (
    ORIGIN_PLAYBACK,
    ORIGIN_RECORDED,
    RECORD_MODE_MIC,
    RECORD_MODE_PLAYBACK,
    CapturePlan,
    LoopbackError,
    _WasapiLoopbackStream,
    find_named_loopback,
    frames_to_mono,
    list_playback_targets,
    open_capture,
    pcm_bytes_to_frames,
    plan_capture,
    playback_capture_available,
    playback_unavailable_reason,
    sounddevice_loopback_kwargs,
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
        "name": "Speakers [Loopback]",
        "max_input_channels": 2,
        "max_output_channels": 0,
        "index": 5,
    },
]


def test_mode_selection_playback_uses_output_not_mic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "votr.loopback.sounddevice_loopback_kwargs",
        lambda: {"loopback": True, "auto_convert": True},
    )
    mic = plan_capture(RECORD_MODE_MIC, devices=DEVICES, preferred_mic="Microphone")
    assert mic.mode == RECORD_MODE_MIC
    assert mic.loopback is False
    assert mic.origin == ORIGIN_RECORDED
    assert mic.backend == "sounddevice"

    play = plan_capture(
        RECORD_MODE_PLAYBACK, devices=DEVICES, preferred_playback="Speakers"
    )
    assert play.mode == RECORD_MODE_PLAYBACK
    assert play.loopback is True
    assert play.origin == ORIGIN_PLAYBACK
    assert play.backend == "sounddevice-wasapi"
    assert play.device == 1
    assert play.device_name == "Speakers"
    assert "icrophone" not in play.device_name.lower()


def test_playback_plan_prefers_named_loopback_when_wasapi_settings_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("votr.loopback.sounddevice_loopback_kwargs", lambda: None)
    play = plan_capture(
        RECORD_MODE_PLAYBACK, devices=DEVICES, preferred_playback="Speakers"
    )
    assert play.backend == "sounddevice-named"
    assert play.device == 5
    assert play.loopback is True
    assert "[Loopback]" in play.device_name


def test_playback_plan_uses_wasapi_on_windows_when_sounddevice_cannot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("votr.loopback.sounddevice_loopback_kwargs", lambda: None)
    monkeypatch.setattr("votr.loopback.sys.platform", "win32")
    no_named = [dev for dev in DEVICES if "[Loopback]" not in str(dev.get("name"))]
    play = plan_capture(
        RECORD_MODE_PLAYBACK, devices=no_named, preferred_playback="Speakers"
    )
    assert play.backend == "wasapi"
    assert play.loopback is True
    assert play.device_name == "Speakers"
    assert play.origin == ORIGIN_PLAYBACK


def test_playback_plan_refuses_mic_only_list() -> None:
    with pytest.raises(LoopbackError, match="playback device"):
        plan_capture(RECORD_MODE_PLAYBACK, devices=[DEVICES[0]])


def test_playback_unavailable_on_linux_without_loopback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("votr.loopback.sounddevice_loopback_kwargs", lambda: None)
    monkeypatch.setattr("votr.loopback.sys.platform", "linux")
    speakers_only = [DEVICES[0], DEVICES[1]]
    reason = playback_unavailable_reason(speakers_only)
    assert "Windows" in reason and "WASAPI" in reason
    assert playback_capture_available(speakers_only) is False
    with pytest.raises(LoopbackError, match="Windows"):
        plan_capture(RECORD_MODE_PLAYBACK, devices=speakers_only)


def test_list_playback_targets_skips_mic_and_cable() -> None:
    names = [dev["name"] for dev in list_playback_targets(DEVICES)]
    assert names == ["Speakers"]
    assert find_named_loopback("Speakers", DEVICES)["index"] == 5
    assert find_named_loopback("Speakers", [DEVICES[0]]) is None


def test_open_playback_passes_loopback_settings_not_mic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}

    class FakeSettings:
        def __init__(self, **kwargs: object) -> None:
            seen["settings"] = kwargs

    class FakeStream:
        def __init__(self, **kwargs: object) -> None:
            seen["stream"] = kwargs

    monkeypatch.setitem(
        sys.modules,
        "sounddevice",
        types.SimpleNamespace(InputStream=FakeStream, WasapiSettings=FakeSettings),
    )
    monkeypatch.setattr(
        "votr.loopback.sounddevice_loopback_kwargs",
        lambda: {"loopback": True, "auto_convert": True},
    )
    plan = CapturePlan(
        mode=RECORD_MODE_PLAYBACK,
        backend="sounddevice-wasapi",
        device=1,
        device_name="Speakers",
        loopback=True,
        origin=ORIGIN_PLAYBACK,
        channels=2,
    )
    open_capture(plan, lambda *_args: None, 48_000, devices=DEVICES)
    stream = seen["stream"]
    settings = seen["settings"]
    assert isinstance(stream, dict) and isinstance(settings, dict)
    assert stream["device"] == 1
    assert stream["extra_settings"] is not None
    assert settings["loopback"] is True


def test_open_playback_refuses_a_mic_plan() -> None:
    sneaky = CapturePlan(
        mode=RECORD_MODE_PLAYBACK,
        backend="sounddevice",
        device=0,
        device_name="Microphone",
        loopback=False,
        origin=ORIGIN_PLAYBACK,
    )
    with pytest.raises(LoopbackError, match="microphone"):
        open_capture(sneaky, lambda *_args: None, 48_000, devices=DEVICES)


def test_wasapi_backend_does_not_open_sounddevice_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened: list[object] = []

    class FakeStream:
        def __init__(self, **kwargs: object) -> None:
            opened.append(kwargs)

    monkeypatch.setitem(
        sys.modules,
        "sounddevice",
        types.SimpleNamespace(InputStream=FakeStream, WasapiSettings=object),
    )
    plan = CapturePlan(
        mode=RECORD_MODE_PLAYBACK,
        backend="wasapi",
        device="Speakers",
        device_name="Speakers",
        loopback=True,
        origin=ORIGIN_PLAYBACK,
        channels=2,
    )
    stream = open_capture(plan, lambda *_args: None, 48_000, devices=DEVICES)
    assert opened == []
    assert isinstance(stream, _WasapiLoopbackStream)


def test_frames_and_pcm_helpers() -> None:
    stereo = np.column_stack(
        [np.ones(4, np.float32), np.full(4, 3.0, np.float32)]
    )
    assert frames_to_mono(stereo).tolist() == [2.0, 2.0, 2.0, 2.0]
    raw = (np.array([0.25, -0.5], np.float32)).tobytes()
    frames = pcm_bytes_to_frames(raw, channels=1, bits=32, is_float=True)
    assert frames.shape == (2, 1)
    assert frames[0, 0] == pytest.approx(0.25)


def test_unknown_mode_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown record mode"):
        plan_capture("both", devices=DEVICES)


def test_sounddevice_loopback_kwargs_honest_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Settings:
        def __init__(self, exclusive: bool = False) -> None:
            self.exclusive = exclusive

    monkeypatch.setitem(
        sys.modules, "sounddevice", types.SimpleNamespace(WasapiSettings=Settings)
    )
    assert sounddevice_loopback_kwargs() is None

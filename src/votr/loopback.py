# Copyright (C) 2026 Michael Cobb
# SPDX-License-Identifier: GPL-3.0-or-later

"""Capture what's playing (playback loopback), never the microphone.

Windows first: sounddevice WASAPI loopback when the build supports it,
else a named ``[Loopback]`` PortAudio device, else a small WASAPI render
loopback via ctypes. Opening a playback plan always requires ``loopback=True``.
"""

from __future__ import annotations

import inspect
import logging
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from votr.live import query_devices
from votr.preview import speaker_devices

log = logging.getLogger("votr.loopback")

RECORD_MODE_MIC = "mic"
RECORD_MODE_PLAYBACK = "playback"
ORIGIN_RECORDED = "recorded"
ORIGIN_PLAYBACK = "playback"

CaptureCallback = Callable[[np.ndarray, int, object, object], None]


class LoopbackError(RuntimeError):
    """Playback capture is unavailable or refused to open a microphone."""


@dataclass(frozen=True)
class CapturePlan:
    """How a clip recording will be opened. Tests inspect this instead of hardware."""

    mode: str
    backend: str
    device: int | str | None
    device_name: str
    loopback: bool
    origin: str
    channels: int = 1


def frames_to_mono(indata: np.ndarray) -> np.ndarray:
    """Copy a sounddevice-style block to mono float32."""
    data = np.asarray(indata, dtype=np.float32)
    if data.ndim == 1:
        return data.copy()
    if data.shape[1] <= 1:
        return data[:, 0].copy()
    return data.mean(axis=1).astype(np.float32)


def is_named_loopback(name: str) -> bool:
    return "[loopback]" in name.lower()


def pcm_bytes_to_frames(
    raw: bytes, *, channels: int, bits: int, is_float: bool
) -> np.ndarray:
    """Decode a WASAPI packet to ``(frames, channels)`` float32."""
    if not raw:
        return np.zeros((0, max(1, channels)), dtype=np.float32)
    if is_float and bits == 32:
        data = np.frombuffer(raw, dtype=np.float32).copy()
    elif bits == 16:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif bits == 32:
        data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise LoopbackError(f"Unsupported mix format: {bits}-bit")
    width = max(1, channels)
    frames = data.size // width
    return data[: frames * width].reshape(frames, width)


def sounddevice_loopback_kwargs() -> dict[str, Any] | None:
    """WasapiSettings kwargs if this sounddevice build can do WASAPI loopback."""
    try:
        import sounddevice as sd
    except Exception:
        return None
    settings = getattr(sd, "WasapiSettings", None)
    if settings is None:
        return None
    try:
        params = inspect.signature(settings).parameters
    except (TypeError, ValueError):
        return None
    if "loopback" not in params:
        return None
    kwargs: dict[str, Any] = {"loopback": True}
    if "auto_convert" in params:
        kwargs["auto_convert"] = True
    return kwargs


def find_named_loopback(
    playback_name: str, devices: list[dict[str, Any]]
) -> dict[str, Any] | None:
    """Input device that is a PortAudio ``[Loopback]`` of ``playback_name``."""
    wanted = playback_name.strip().lower()
    matches: list[dict[str, Any]] = []
    for device in devices:
        name = str(device.get("name", ""))
        if not is_named_loopback(name):
            continue
        if int(device.get("max_input_channels", 0)) <= 0:
            continue
        matches.append(device)
    if not matches:
        return None
    if wanted:
        for device in matches:
            base = str(device.get("name", "")).lower().replace("[loopback]", "").strip()
            if wanted in base or base in wanted:
                return device
    return matches[0]


def list_playback_targets(
    devices: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Playback (output) devices that can be looped back — not microphones."""
    queried = devices is None
    found = query_devices() if queried else devices
    speakers = speaker_devices(found)
    if speakers:
        return speakers
    if queried and sys.platform == "win32":
        return [
            {
                "name": "Default playback device",
                "index": None,
                "max_output_channels": 2,
                "max_input_channels": 0,
            }
        ]
    return []


def playback_unavailable_reason(
    devices: list[dict[str, Any]] | None = None,
) -> str:
    if sys.platform == "win32":
        if devices is not None and not list_playback_targets(devices):
            return (
                "No playback device found — pick speakers in Settings, or upload "
                "a clip."
            )
        return ""
    found = devices if devices is not None else query_devices()
    if sounddevice_loopback_kwargs() and list_playback_targets(found):
        return ""
    if any(is_named_loopback(str(dev.get("name", ""))) for dev in found):
        return ""
    return (
        "Recording what's playing needs Windows (WASAPI loopback). This computer "
        "can't capture speaker audio — upload the video's audio instead."
    )


def playback_capture_available(
    devices: list[dict[str, Any]] | None = None,
) -> bool:
    return playback_unavailable_reason(devices) == ""


def _preferred_playback(
    targets: list[dict[str, Any]], preferred: str
) -> dict[str, Any] | None:
    if preferred:
        folded = preferred.lower()
        for device in targets:
            if str(device.get("name", "")).lower() == folded:
                return device
        for device in targets:
            name = str(device.get("name", "")).lower()
            if folded in name or name in folded:
                return device
    return targets[0] if targets else None


def _channels_for(device: dict[str, Any], *, loopback_named: bool) -> int:
    if loopback_named:
        return max(1, int(device.get("max_input_channels", 0) or 1))
    return max(1, int(device.get("max_output_channels", 0) or 2))


def _assert_not_microphone(plan: CapturePlan, devices: list[dict[str, Any]]) -> None:
    """Refuse a plan that would open a normal capture (mic) device."""
    if plan.mode != RECORD_MODE_PLAYBACK or not plan.loopback:
        raise LoopbackError(
            "Refusing to open the microphone for “what's playing”."
        )
    if plan.backend == "wasapi":
        return
    if plan.backend == "sounddevice-named":
        if not is_named_loopback(plan.device_name):
            raise LoopbackError(
                "Named loopback device is missing [Loopback] — not opening the mic."
            )
        return
    if plan.backend == "sounddevice-wasapi":
        for device in devices:
            if device.get("index") == plan.device or str(device.get("name", "")) == str(
                plan.device
            ):
                if int(device.get("max_output_channels", 0)) <= 0:
                    raise LoopbackError(
                        "WASAPI loopback must open a playback device, not the mic."
                    )
                return
        return
    raise LoopbackError(f"Unknown playback backend {plan.backend!r}.")


def plan_capture(
    mode: str,
    *,
    devices: list[dict[str, Any]] | None = None,
    preferred_playback: str = "",
    preferred_mic: str = "",
) -> CapturePlan:
    """Pick a capture path. Playback never selects a microphone."""
    if mode == RECORD_MODE_MIC:
        name = preferred_mic.strip() or "microphone"
        return CapturePlan(
            mode=RECORD_MODE_MIC,
            backend="sounddevice",
            device=None,
            device_name=name,
            loopback=False,
            origin=ORIGIN_RECORDED,
            channels=1,
        )
    if mode != RECORD_MODE_PLAYBACK:
        raise ValueError(f"Unknown record mode {mode!r}")

    found = devices if devices is not None else query_devices()
    targets = list_playback_targets(found)
    target = _preferred_playback(targets, preferred_playback)
    if target is None:
        raise LoopbackError(
            "No playback device found — pick speakers in Settings, or upload a clip."
        )

    name = str(target.get("name", "playback"))
    index = target.get("index")
    sd_loopback = sounddevice_loopback_kwargs()
    if sd_loopback is not None:
        plan = CapturePlan(
            mode=RECORD_MODE_PLAYBACK,
            backend="sounddevice-wasapi",
            device=index if index is not None else name,
            device_name=name,
            loopback=True,
            origin=ORIGIN_PLAYBACK,
            channels=_channels_for(target, loopback_named=False),
        )
        _assert_not_microphone(plan, found)
        return plan

    named = find_named_loopback(name, found)
    if named is not None:
        plan = CapturePlan(
            mode=RECORD_MODE_PLAYBACK,
            backend="sounddevice-named",
            device=named.get("index", named.get("name")),
            device_name=str(named.get("name", name)),
            loopback=True,
            origin=ORIGIN_PLAYBACK,
            channels=_channels_for(named, loopback_named=True),
        )
        _assert_not_microphone(plan, found)
        return plan

    if sys.platform == "win32":
        plan = CapturePlan(
            mode=RECORD_MODE_PLAYBACK,
            backend="wasapi",
            device=None if name == "Default playback device" else name,
            device_name=name,
            loopback=True,
            origin=ORIGIN_PLAYBACK,
            channels=_channels_for(target, loopback_named=False),
        )
        _assert_not_microphone(plan, found)
        return plan

    raise LoopbackError(playback_unavailable_reason(found))


def open_capture(
    plan: CapturePlan,
    callback: CaptureCallback,
    samplerate: int,
    *,
    devices: list[dict[str, Any]] | None = None,
) -> Any:
    """Open the stream described by ``plan``. Playback refuses a mic path."""
    if plan.mode == RECORD_MODE_MIC:
        return _open_sounddevice_input(
            device=plan.device,
            channels=1,
            samplerate=samplerate,
            callback=callback,
            extra_settings=None,
        )
    found = devices if devices is not None else query_devices()
    _assert_not_microphone(plan, found)
    if plan.backend == "wasapi":
        return _WasapiLoopbackStream(
            device_name="" if plan.device is None else str(plan.device),
            samplerate=samplerate,
            callback=callback,
        )
    extra = None
    if plan.backend == "sounddevice-wasapi":
        kwargs = sounddevice_loopback_kwargs()
        if not kwargs:
            raise LoopbackError(
                "sounddevice WASAPI loopback is not available on this build."
            )
        import sounddevice as sd

        extra = sd.WasapiSettings(**kwargs)
    return _open_sounddevice_input(
        device=plan.device,
        channels=plan.channels,
        samplerate=samplerate,
        callback=callback,
        extra_settings=extra,
    )


def _open_sounddevice_input(
    *,
    device: int | str | None,
    channels: int,
    samplerate: int,
    callback: CaptureCallback,
    extra_settings: Any,
) -> Any:
    import sounddevice as sd

    return sd.InputStream(
        samplerate=samplerate,
        channels=max(1, channels),
        dtype="float32",
        device=device,
        extra_settings=extra_settings,
        callback=callback,
    )


# -- Windows WASAPI render loopback (ctypes; imported only on Windows) ------


_AUDCLNT_STREAMFLAGS_LOOPBACK = 0x00020000
_AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM = 0x80000000
_AUDCLNT_STREAMFLAGS_SRC_DEFAULT_QUALITY = 0x08000000
_AUDCLNT_BUFFERFLAGS_SILENT = 0x1
_E_RENDER = 0
_E_CONSOLE = 0
_DEVICE_STATE_ACTIVE = 0x1
_CLSCTX_ALL = 23
_STGM_READ = 0
_COINIT_MULTITHREADED = 0
_RPC_E_CHANGED_MODE = 0x80010106
_VT_LPWSTR = 31
_REFTIMES_PER_SEC = 10_000_000
_CLSID_MMDEV = "{BCDE0395-E52F-467C-8E3D-C4579291692E}"
_IID_ENUM = "{A95664D2-9614-4F35-A746-DE8DB63617E6}"
_IID_CLIENT = "{1CB9AD4C-DBFA-4c32-B178-C2F568A703B2}"
_IID_CAPTURE = "{C8ADBD64-E71E-48a0-A4DE-185C395CD317}"
_PKEY_NAME = "{A45C254E-DF1C-4EFD-8020-67D146A850E0}"


def _windows_modules() -> tuple[Any, Any]:
    if sys.platform != "win32":
        raise LoopbackError("WASAPI loopback is only available on Windows.")
    import ctypes

    return ctypes, ctypes.windll.ole32


def _com_ok(hr: int, what: str) -> None:
    if hr == 0:
        return
    unsigned = hr + 2**32 if hr < 0 else hr
    raise LoopbackError(f"{what} failed (HRESULT 0x{unsigned:08X}).")


def _guid_type(ctypes: Any) -> Any:
    class GUID(ctypes.Structure):
        _fields_ = [
            ("Data1", ctypes.c_uint32),
            ("Data2", ctypes.c_uint16),
            ("Data3", ctypes.c_uint16),
            ("Data4", ctypes.c_ubyte * 8),
        ]

    return GUID


def _guid(ctypes: Any, ole32: Any, text: str) -> Any:
    guid_cls = _guid_type(ctypes)
    guid = guid_cls()
    ole32.IIDFromString.argtypes = [ctypes.c_wchar_p, ctypes.POINTER(guid_cls)]
    ole32.IIDFromString.restype = ctypes.HRESULT
    _com_ok(ole32.IIDFromString(text, ctypes.byref(guid)), "IIDFromString")
    return guid


def _as_void(ctypes: Any, obj: Any) -> Any:
    if isinstance(obj, ctypes.c_void_p):
        return obj
    return ctypes.c_void_p(obj)


def _vtable(ctypes: Any, obj: Any, index: int, restype: Any, *argtypes: Any) -> Any:
    iface = _as_void(ctypes, obj)
    vtbl_ptr = ctypes.cast(iface, ctypes.POINTER(ctypes.c_void_p))[0]
    vtbl = ctypes.cast(vtbl_ptr, ctypes.POINTER(ctypes.c_void_p))
    func = ctypes.WINFUNCTYPE(restype, ctypes.c_void_p, *argtypes)(vtbl[index])
    return lambda *args: func(iface, *args)


def _release(ctypes: Any, obj: Any) -> None:
    if not obj:
        return
    try:
        _vtable(ctypes, obj, 2, ctypes.c_ulong)()
    except Exception:  # noqa: BLE001 — teardown must not raise
        return


class _WasapiLoopbackStream:
    """Capture the mix playing on a render endpoint (AUDCLNT_STREAMFLAGS_LOOPBACK)."""

    def __init__(
        self,
        *,
        device_name: str,
        samplerate: int,
        callback: CaptureCallback,
    ) -> None:
        self._device_name = device_name
        self._samplerate = samplerate
        self._callback = callback
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None
        self._started = threading.Event()

    def start(self) -> None:
        if sys.platform != "win32":
            raise LoopbackError("WASAPI loopback is only available on Windows.")
        self._stop.clear()
        self._error = None
        self._started.clear()
        self._thread = threading.Thread(
            target=self._run, name="votr-wasapi-loopback", daemon=True
        )
        self._thread.start()
        if not self._started.wait(timeout=5.0):
            self.stop()
            raise LoopbackError(str(self._error or "WASAPI loopback failed to start."))
        if self._error is not None:
            self.stop()
            raise LoopbackError(str(self._error))

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def close(self) -> None:
        self.stop()

    def _run(self) -> None:
        ctypes, ole32 = _windows_modules()
        enumerator = None
        device = None
        client = None
        capture = None
        mix = None
        try:
            _init_com(ctypes, ole32)
            enumerator = _create_enumerator(ctypes, ole32)
            device = _resolve_render_device(
                ctypes, ole32, enumerator, self._device_name
            )
            client, mix, channels, bits, is_float, rate = _init_loopback_client(
                ctypes, ole32, device
            )
            capture = _get_capture_client(ctypes, ole32, client)
            start = _vtable(ctypes, client, 10, ctypes.HRESULT)
            _com_ok(start(), "IAudioClient.Start")
            self._started.set()
            next_size = _vtable(
                ctypes, capture, 5, ctypes.HRESULT, ctypes.POINTER(ctypes.c_uint32)
            )
            get_buffer = _vtable(
                ctypes,
                capture,
                3,
                ctypes.HRESULT,
                ctypes.POINTER(ctypes.POINTER(ctypes.c_byte)),
                ctypes.POINTER(ctypes.c_uint32),
                ctypes.POINTER(ctypes.c_uint32),
                ctypes.c_void_p,
                ctypes.c_void_p,
            )
            release = _vtable(ctypes, capture, 4, ctypes.HRESULT, ctypes.c_uint32)
            while not self._stop.is_set():
                packet = ctypes.c_uint32(0)
                _com_ok(next_size(ctypes.byref(packet)), "GetNextPacketSize")
                if packet.value == 0:
                    time.sleep(0.01)
                    continue
                data_ptr = ctypes.POINTER(ctypes.c_byte)()
                frames = ctypes.c_uint32(0)
                flags = ctypes.c_uint32(0)
                _com_ok(
                    get_buffer(
                        ctypes.byref(data_ptr),
                        ctypes.byref(frames),
                        ctypes.byref(flags),
                        None,
                        None,
                    ),
                    "GetBuffer",
                )
                nframes = int(frames.value)
                if nframes > 0 and data_ptr:
                    nbytes = nframes * channels * max(1, bits // 8)
                    raw = ctypes.string_at(data_ptr, nbytes)
                    block = pcm_bytes_to_frames(
                        raw, channels=channels, bits=bits, is_float=is_float
                    )
                    if flags.value & _AUDCLNT_BUFFERFLAGS_SILENT:
                        block.fill(0)
                    if rate and rate != self._samplerate:
                        block = _resample_block(block, rate, self._samplerate)
                    self._callback(block, block.shape[0], None, None)
                _com_ok(release(nframes), "ReleaseBuffer")
        except BaseException as exc:  # noqa: BLE001 — surface to start()
            self._error = exc
            self._started.set()
        finally:
            if client is not None:
                try:
                    _vtable(ctypes, client, 11, ctypes.HRESULT)()
                except Exception:  # noqa: BLE001
                    pass
            if mix is not None:
                try:
                    ole32.CoTaskMemFree(mix)
                except Exception:  # noqa: BLE001
                    pass
            _release(ctypes, capture)
            _release(ctypes, client)
            _release(ctypes, device)
            _release(ctypes, enumerator)


def _init_com(ctypes: Any, ole32: Any) -> None:
    ole32.CoInitializeEx.argtypes = [ctypes.c_void_p, ctypes.c_uint]
    ole32.CoInitializeEx.restype = ctypes.HRESULT
    hr = ole32.CoInitializeEx(None, _COINIT_MULTITHREADED)
    unsigned = hr + 2**32 if hr < 0 else hr
    if hr not in (0, 1) and unsigned != _RPC_E_CHANGED_MODE:
        _com_ok(hr, "CoInitializeEx")


def _create_enumerator(ctypes: Any, ole32: Any) -> Any:
    clsid = _guid(ctypes, ole32, _CLSID_MMDEV)
    iid = _guid(ctypes, ole32, _IID_ENUM)
    ptr = ctypes.c_void_p()
    ole32.CoCreateInstance.restype = ctypes.HRESULT
    _com_ok(
        ole32.CoCreateInstance(
            ctypes.byref(clsid),
            None,
            _CLSCTX_ALL,
            ctypes.byref(iid),
            ctypes.byref(ptr),
        ),
        "CoCreateInstance(IMMDeviceEnumerator)",
    )
    return ptr.value


def _resolve_render_device(
    ctypes: Any, ole32: Any, enumerator: Any, name: str
) -> Any:
    wanted = name.strip()
    if not wanted:
        return _default_render(ctypes, enumerator)
    collection = ctypes.c_void_p()
    enum_ep = _vtable(
        ctypes,
        enumerator,
        3,
        ctypes.HRESULT,
        ctypes.c_int,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_void_p),
    )
    _com_ok(
        enum_ep(_E_RENDER, _DEVICE_STATE_ACTIVE, ctypes.byref(collection)),
        "EnumAudioEndpoints",
    )
    try:
        get_count = _vtable(
            ctypes, collection.value, 3, ctypes.HRESULT, ctypes.POINTER(ctypes.c_uint32)
        )
        item = _vtable(
            ctypes,
            collection.value,
            4,
            ctypes.HRESULT,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_void_p),
        )
        count = ctypes.c_uint32(0)
        _com_ok(get_count(ctypes.byref(count)), "GetCount")
        folded = wanted.lower()
        for index in range(int(count.value)):
            device = ctypes.c_void_p()
            _com_ok(item(index, ctypes.byref(device)), "IMMDeviceCollection.Item")
            friendly = _device_friendly_name(ctypes, ole32, device.value)
            if folded == friendly.lower() or folded in friendly.lower():
                return device.value
            _release(ctypes, device.value)
    finally:
        _release(ctypes, collection.value)
    log.info("No render device named %r; using the default output.", wanted)
    return _default_render(ctypes, enumerator)


def _default_render(ctypes: Any, enumerator: Any) -> Any:
    device = ctypes.c_void_p()
    get_default = _vtable(
        ctypes,
        enumerator,
        4,
        ctypes.HRESULT,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_void_p),
    )
    _com_ok(
        get_default(_E_RENDER, _E_CONSOLE, ctypes.byref(device)),
        "GetDefaultAudioEndpoint(eRender)",
    )
    return device.value


def _device_friendly_name(ctypes: Any, ole32: Any, device: Any) -> str:
    store = ctypes.c_void_p()
    open_store = _vtable(
        ctypes,
        device,
        4,
        ctypes.HRESULT,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_void_p),
    )
    _com_ok(open_store(_STGM_READ, ctypes.byref(store)), "OpenPropertyStore")
    guid_cls = _guid_type(ctypes)

    class PKEY(ctypes.Structure):
        _fields_ = [("fmtid", guid_cls), ("pid", ctypes.c_uint32)]

    class PROPVARIANT(ctypes.Structure):
        _fields_ = [
            ("vt", ctypes.c_uint16),
            ("wReserved1", ctypes.c_uint16),
            ("wReserved2", ctypes.c_uint16),
            ("wReserved3", ctypes.c_uint16),
            ("data", ctypes.c_void_p),
        ]

    try:
        key = PKEY(_guid(ctypes, ole32, _PKEY_NAME), 14)
        value = PROPVARIANT()
        get_value = _vtable(
            ctypes,
            store.value,
            5,
            ctypes.HRESULT,
            ctypes.POINTER(PKEY),
            ctypes.POINTER(PROPVARIANT),
        )
        _com_ok(get_value(ctypes.byref(key), ctypes.byref(value)), "GetValue")
        try:
            if value.vt != _VT_LPWSTR or not value.data:
                return ""
            return ctypes.wstring_at(value.data)
        finally:
            ole32.PropVariantClear.argtypes = [ctypes.POINTER(PROPVARIANT)]
            ole32.PropVariantClear(ctypes.byref(value))
    finally:
        _release(ctypes, store.value)


def _waveformat_type(ctypes: Any) -> Any:
    class WAVEFORMATEX(ctypes.Structure):
        _fields_ = [
            ("wFormatTag", ctypes.c_uint16),
            ("nChannels", ctypes.c_uint16),
            ("nSamplesPerSec", ctypes.c_uint32),
            ("nAvgBytesPerSec", ctypes.c_uint32),
            ("nBlockAlign", ctypes.c_uint16),
            ("wBitsPerSample", ctypes.c_uint16),
            ("cbSize", ctypes.c_uint16),
        ]

    return WAVEFORMATEX


def _init_loopback_client(
    ctypes: Any, ole32: Any, device: Any
) -> tuple[Any, Any, int, int, bool, int]:
    wave_cls = _waveformat_type(ctypes)
    iid_client = _guid(ctypes, ole32, _IID_CLIENT)
    client = ctypes.c_void_p()
    activate = _vtable(
        ctypes,
        device,
        3,
        ctypes.HRESULT,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
    )
    _com_ok(
        activate(ctypes.byref(iid_client), _CLSCTX_ALL, None, ctypes.byref(client)),
        "IMMDevice.Activate(IAudioClient)",
    )
    mix_ptr = ctypes.POINTER(wave_cls)()
    get_mix = _vtable(
        ctypes,
        client.value,
        8,
        ctypes.HRESULT,
        ctypes.POINTER(ctypes.POINTER(wave_cls)),
    )
    _com_ok(get_mix(ctypes.byref(mix_ptr)), "GetMixFormat")
    mix = mix_ptr.contents
    channels = int(mix.nChannels)
    bits = int(mix.wBitsPerSample)
    rate = int(mix.nSamplesPerSec)
    is_float = mix.wFormatTag in (3, 0xFFFE)
    flags = (
        _AUDCLNT_STREAMFLAGS_LOOPBACK
        | _AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM
        | _AUDCLNT_STREAMFLAGS_SRC_DEFAULT_QUALITY
    )
    initialize = _vtable(
        ctypes,
        client.value,
        3,
        ctypes.HRESULT,
        ctypes.c_int,
        ctypes.c_uint32,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.POINTER(wave_cls),
        ctypes.c_void_p,
    )
    hr = initialize(0, flags, _REFTIMES_PER_SEC // 10, 0, mix_ptr, None)
    if hr != 0:
        _com_ok(
            initialize(
                0,
                _AUDCLNT_STREAMFLAGS_LOOPBACK,
                _REFTIMES_PER_SEC // 10,
                0,
                mix_ptr,
                None,
            ),
            "IAudioClient.Initialize(LOOPBACK)",
        )
    return client.value, mix_ptr, channels, bits, is_float, rate


def _get_capture_client(ctypes: Any, ole32: Any, client: Any) -> Any:
    iid = _guid(ctypes, ole32, _IID_CAPTURE)
    capture = ctypes.c_void_p()
    get_service = _vtable(
        ctypes,
        client,
        14,
        ctypes.HRESULT,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
    )
    _com_ok(
        get_service(ctypes.byref(iid), ctypes.byref(capture)),
        "IAudioClient.GetService(IAudioCaptureClient)",
    )
    return capture.value


def _resample_block(block: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate or block.size == 0:
        return block
    mono = frames_to_mono(block)
    count = int(round(mono.size * dst_rate / src_rate))
    positions = np.linspace(0.0, mono.size - 1, count)
    out = np.interp(positions, np.arange(mono.size), mono).astype(np.float32)
    return out.reshape(-1, 1)

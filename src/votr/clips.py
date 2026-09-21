# Copyright (C) 2026 Michael Cobb
# SPDX-License-Identifier: GPL-3.0-or-later

"""Mimic clip library: named reference recordings for Neural Voices.

Each clip is a mono 48 kHz WAV under ``<data>/clips/<id>.wav`` with a sidecar
``<id>.json`` (name, origin, seconds, created). Voices refer to a clip by id
(``reference_clip_id``) and, for the Engine, by absolute path
(``reference_clip``); the id wins if the data folder ever moves.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np

from votr.wavutil import read_wav, write_wav

log = logging.getLogger("votr.clips")

CLIP_SAMPLE_RATE = 48_000
MAX_CLIP_SECONDS = 30.0
MIN_CLIP_SECONDS = 1.0
CLIP_TARGET_PEAK = 0.9
QUIET_PEAK_DB = -24.0
IMPORT_EXTENSIONS = (".wav", ".flac", ".mp3", ".ogg", ".aiff", ".aif", ".m4a")
REFERENCE_CLIP_ID_KEY = "reference_clip_id"
ORIGIN_RECORDED = "recorded"
ORIGIN_PLAYBACK = "playback"


class ClipError(ValueError):
    """A clip could not be added (too short, unreadable, unsupported)."""


@dataclass
class Clip:
    id: str
    name: str
    seconds: float
    origin: str = ""
    created: str = ""
    source_peak_db: float = 0.0

    @property
    def label(self) -> str:
        return f"{self.name} ({self.seconds:.1f} s)"

    @property
    def was_quiet(self) -> bool:
        return self.source_peak_db < QUIET_PEAK_DB

    @property
    def level_note(self) -> str:
        note = f"recorded peak {self.source_peak_db:.0f} dBFS"
        if self.was_quiet:
            if self.origin == ORIGIN_PLAYBACK:
                note += (
                    " — quiet. Stored as recorded (not boosted); turn the "
                    "video up for a cleaner mimic"
                )
            else:
                note += (
                    " — quiet. Stored as recorded (not boosted); get closer "
                    "to the mic or raise Mic gain"
                )
        return note


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def decode_audio(path: Path) -> tuple[np.ndarray, int]:
    """Mono float32 at 48 kHz from any format Pedalboard can open."""
    path = Path(path)
    if path.suffix.lower() == ".wav":
        try:
            samples, rate = read_wav(path)
            if rate == CLIP_SAMPLE_RATE:
                return samples, rate
        except Exception:  # noqa: BLE001 — fall through to the full decoder
            pass
    try:
        from pedalboard.io import AudioFile

        with AudioFile(str(path)) as source:
            reader = source.resampled_to(CLIP_SAMPLE_RATE)
            frames = reader.read(reader.frames)
    except Exception as exc:
        raise ClipError(f"Could not read {path.name}: {exc}") from exc
    audio = np.asarray(frames, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=0)
    return audio.reshape(-1), CLIP_SAMPLE_RATE


def _trim_silence(samples: np.ndarray, threshold: float = 0.003) -> np.ndarray:
    loud = np.where(np.abs(samples) > threshold)[0]
    if loud.size == 0:
        return samples[:0]
    pad = int(0.1 * CLIP_SAMPLE_RATE)
    start = max(0, int(loud[0]) - pad)
    end = min(samples.size, int(loud[-1]) + pad)
    return samples[start:end]


class ClipLibrary:
    def __init__(self, data_dir: Path) -> None:
        self.root = Path(data_dir) / "clips"
        self.root.mkdir(parents=True, exist_ok=True)

    # -- storage --------------------------------------------------------------

    def path_for(self, clip_id: str) -> Path:
        return self.root / f"{clip_id}.wav"

    def _meta_path(self, clip_id: str) -> Path:
        return self.root / f"{clip_id}.json"

    def _write_meta(self, clip: Clip) -> None:
        self._meta_path(clip.id).write_text(
            json.dumps(asdict(clip), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def list(self) -> list[Clip]:
        clips: list[Clip] = []
        for meta in sorted(self.root.glob("*.json")):
            try:
                data = json.loads(meta.read_text(encoding="utf-8"))
                clip = Clip(
                    id=str(data["id"]),
                    name=str(data.get("name") or meta.stem),
                    seconds=float(data.get("seconds") or 0.0),
                    origin=str(data.get("origin") or ""),
                    created=str(data.get("created") or ""),
                    source_peak_db=float(data.get("source_peak_db") or 0.0),
                )
            except (OSError, ValueError, KeyError, TypeError) as exc:
                log.warning("Skipped clip metadata %s: %s", meta.name, exc)
                continue
            if self.path_for(clip.id).is_file():
                clips.append(clip)
        clips.sort(key=lambda clip: clip.name.lower())
        return clips

    def get(self, clip_id: str) -> Clip | None:
        return next((clip for clip in self.list() if clip.id == clip_id), None)

    def by_name(self, name: str) -> Clip | None:
        wanted = name.strip().lower()
        return next((clip for clip in self.list() if clip.name.lower() == wanted), None)

    # -- adding ---------------------------------------------------------------

    def add(
        self,
        samples: np.ndarray,
        sample_rate: int,
        name: str,
        *,
        origin: str = ORIGIN_RECORDED,
    ) -> Clip:
        audio = np.asarray(samples, dtype=np.float32).reshape(-1)
        if sample_rate != CLIP_SAMPLE_RATE:
            positions = np.linspace(
                0.0,
                audio.size - 1,
                int(round(audio.size * CLIP_SAMPLE_RATE / sample_rate)),
            )
            audio = np.interp(positions, np.arange(audio.size), audio).astype(
                np.float32
            )
        audio = _trim_silence(audio)
        audio = audio[: int(MAX_CLIP_SECONDS * CLIP_SAMPLE_RATE)]
        seconds = audio.size / CLIP_SAMPLE_RATE
        if seconds < MIN_CLIP_SECONDS:
            raise ClipError(
                f"Clip is {seconds:.1f} s of sound; a mimic clip needs at least "
                f"{MIN_CLIP_SECONDS:.0f} s (10–30 s of clear speech is better)."
            )
        peak = float(np.max(np.abs(audio)))
        source_peak_db = 20.0 * np.log10(max(peak, 1e-6))
        # Peak-limit hot clips so playback does not crackle. Do not boost
        # quiet ones: that raises the noise floor and washes identity. X-VC
        # has its own volume_normalize for the model.
        if peak > CLIP_TARGET_PEAK:
            audio = audio * (CLIP_TARGET_PEAK / peak)
        clip = Clip(
            id=str(uuid4()),
            name=self.unique_name(name.strip() or "Mimic clip"),
            seconds=round(seconds, 2),
            origin=origin,
            created=_now(),
            source_peak_db=round(float(source_peak_db), 1),
        )
        write_wav(self.path_for(clip.id), audio, CLIP_SAMPLE_RATE)
        self._write_meta(clip)
        return clip

    def import_file(self, path: Path, name: str | None = None) -> Clip:
        path = Path(path)
        if path.suffix.lower() not in IMPORT_EXTENSIONS:
            raise ClipError(
                f"{path.suffix or path.name} is not a supported audio file "
                f"({', '.join(IMPORT_EXTENSIONS)})."
            )
        samples, rate = decode_audio(path)
        return self.add(samples, rate, name or path.stem, origin=f"file:{path.name}")

    def unique_name(self, name: str) -> str:
        taken = {clip.name.lower() for clip in self.list()}
        if name.lower() not in taken:
            return name
        number = 2
        while f"{name} {number}".lower() in taken:
            number += 1
        return f"{name} {number}"

    # -- editing --------------------------------------------------------------

    def rename(self, clip_id: str, name: str) -> Clip | None:
        clip = self.get(clip_id)
        if clip is None:
            return None
        wanted = name.strip()
        if not wanted or wanted.lower() == clip.name.lower():
            return clip
        clip.name = self.unique_name(wanted)
        self._write_meta(clip)
        return clip

    def delete(self, clip_id: str) -> None:
        for path in (self.path_for(clip_id), self._meta_path(clip_id)):
            if path.exists():
                path.unlink()

    def load(self, clip_id: str) -> tuple[np.ndarray, int] | None:
        path = self.path_for(clip_id)
        if not path.is_file():
            return None
        return read_wav(path)

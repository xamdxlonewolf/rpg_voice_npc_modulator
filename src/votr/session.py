# Copyright (C) 2026 Michael Cobb
# SPDX-License-Identifier: GPL-3.0-or-later

"""In-memory session: Voices, Active Voice, DSP Engine."""

from __future__ import annotations

from pathlib import Path

from votr.devices import DeviceSettings
from votr.dsp import DspEngine
from votr.latency import (
    LatencyProbe,
    block_for_quality,
    choose_probe,
    quality_for_block,
)
from votr.latency import run_latency_test as probe_latency
from votr.live import AudioDeviceError
from votr.macros import load_macros
from votr.roleplay import RoleplayError, RoleplayPath
from votr.store import VoiceStore
from votr.voice import DSP_ENGINE_ID, Voice

INSTALLED_ENGINE_IDS = {DSP_ENGINE_ID, "passthrough"}


class Session:
    def __init__(self, data_dir: Path | None = None) -> None:
        self.store = VoiceStore(data_dir)
        self.macros = load_macros()
        self.settings = DeviceSettings.load(self.store.root.parent)
        block = self.settings.block_size or None
        self.engine = DspEngine(macros=self.macros, block_size=block)
        self.path = RoleplayPath(self.engine, self.settings)
        self.voices = self.store.load_all()
        self.warnings = list(self.store.warnings)
        self.draft = Voice.new()
        self.active_id: str | None = None
        self.roleplay_on = False
        self.take: object | None = None
        self.auto_replay = True
        self._saved = self.draft.to_dict()

    def is_dirty(self) -> bool:
        return self.draft.to_dict() != self._saved

    def engine_installed(self, engine_id: str) -> bool:
        return engine_id in INSTALLED_ENGINE_IDS

    def voice_by_id(self, voice_id: str) -> Voice | None:
        return next((voice for voice in self.voices if voice.id == voice_id), None)

    def replace_voice(self, voice: Voice) -> None:
        self.voices = [item for item in self.voices if item.id != voice.id]
        self.voices.append(voice)

    def apply_draft_to_engine(self) -> None:
        self.engine.set_params(self.draft.params)

    def apply_tag(self, tag: str) -> None:
        if tag not in self.macros:
            return
        if tag not in self.draft.tone_tags:
            self.draft.tone_tags.append(tag)
        self.engine.apply_macro(tag)
        self.draft.params.update(self.macros[tag].params)

    def save_draft(self) -> Voice:
        if not self.draft.name.strip():
            self.draft.name = "Untitled Voice"
        saved = Voice.from_dict(self.draft.to_dict())
        self.store.save(saved)
        self.replace_voice(saved)
        self.draft = Voice.from_dict(saved.to_dict())
        self._saved = self.draft.to_dict()
        if self.active_id is None:
            self.set_active(saved.id)
        return saved

    def save_draft_as_new(self) -> Voice:
        self.draft = self.draft.duplicate()
        return self.save_draft()

    def delete_voice(self, voice_id: str) -> None:
        self.store.delete(voice_id)
        self.voices = [voice for voice in self.voices if voice.id != voice_id]
        if self.active_id == voice_id:
            self.active_id = self.voices[0].id if self.voices else None
        if self.draft.id == voice_id:
            self.edit_new()

    def delete_draft(self) -> None:
        self.delete_voice(self.draft.id)

    def duplicate_voice(self, voice: Voice) -> Voice:
        copy = voice.duplicate()
        self.store.save(copy)
        self.replace_voice(copy)
        return copy

    def edit(self, voice: Voice) -> None:
        self.draft = Voice.from_dict(voice.to_dict())
        self._saved = self.draft.to_dict()
        self.apply_draft_to_engine()

    def edit_new(self) -> None:
        self.draft = Voice.new()
        self._saved = self.draft.to_dict()
        defaults = {spec.key: spec.default for spec in self.engine.parameter_schema()}
        self.engine.set_params(defaults)

    def set_active(self, voice_id: str) -> Voice | None:
        voice = self.voice_by_id(voice_id)
        if voice is None or not self.engine_installed(voice.engine_id):
            return None
        voice.touch(used=True)
        self.store.save(voice)
        self.active_id = voice.id
        fade = 30.0 if self.roleplay_on else 0.0
        self.engine.set_params(voice.params, crossfade_ms=fade)
        return voice

    def active_voice(self) -> Voice | None:
        if self.active_id is None:
            return None
        return self.voice_by_id(self.active_id)

    def start_roleplay(self) -> None:
        self.path.start()
        self.roleplay_on = True

    def stop_roleplay(self) -> None:
        self.path.stop()
        self.roleplay_on = False

    def save_settings(self) -> None:
        self.settings.save(self.store.root.parent)

    def apply_block_size(self, block_size: int) -> None:
        if block_size == self.engine.block_size:
            self.settings.block_size = block_size
            self.settings.latency_quality = quality_for_block(block_size)
            return
        was_on = self.roleplay_on
        if was_on:
            self.stop_roleplay()
        params = self.engine.params()
        prefer = bool(getattr(self.engine, "_use_rubband", False))
        self.engine = DspEngine(
            macros=self.macros, block_size=block_size, prefer_rubband=prefer
        )
        self.engine.set_params(params)
        self.path.engine = self.engine
        self.settings.block_size = block_size
        self.settings.latency_quality = quality_for_block(block_size)
        if was_on:
            try:
                self.start_roleplay()
            except (RoleplayError, AudioDeviceError):
                self.roleplay_on = False

    def apply_latency_quality(self, quality: int) -> None:
        self.apply_block_size(block_for_quality(quality))
        self.save_settings()

    def run_latency_test(self, *, duration_s: float = 10.0) -> LatencyProbe:
        prefer = bool(getattr(self.engine, "_use_rubband", False))

        def make(block: int) -> DspEngine:
            return DspEngine(
                macros=self.macros, block_size=block, prefer_rubband=prefer
            )

        probes = probe_latency(make, duration_s=duration_s)
        chosen = choose_probe(probes)
        self.settings.latency_method = chosen.method
        self.settings.latency_ms = chosen.measured_ms
        self.settings.latency_engine_ms = chosen.engine_ms
        self.settings.latency_device_ms = chosen.device_ms
        self.settings.latency_blocked = chosen.blocked or ""
        if chosen.total_ms is not None:
            self.apply_block_size(chosen.block_size)
        self.save_settings()
        return chosen

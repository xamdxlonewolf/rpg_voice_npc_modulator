# Copyright (C) 2026 Michael Cobb
# SPDX-License-Identifier: GPL-3.0-or-later

"""In-memory session: Voices, Active Voice, DSP Engine."""

from __future__ import annotations

import logging
import threading
from pathlib import Path

from votr.clips import REFERENCE_CLIP_ID_KEY, Clip, ClipLibrary
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
from votr.neural import (
    NEURAL_ENGINE_ID,
    NeuralRuntime,
    build_neural_engine,
    probe_runtime,
)
from votr.neural_engine import REFERENCE_CLIP_KEY, NeuralEngine, NeuralUnavailable
from votr.presets import load_presets, preset_by_name, voice_from_preset
from votr.roleplay import RoleplayError, RoleplayPath
from votr.store import VoiceStore
from votr.voice import DSP_ENGINE_ID, Voice
from votr.voicedesign import VoiceDesign, design_voice

INSTALLED_ENGINE_IDS = {DSP_ENGINE_ID, "passthrough"}
log = logging.getLogger("votr.session")


class Session:
    def __init__(self, data_dir: Path | None = None) -> None:
        self.store = VoiceStore(data_dir)
        self.clips = ClipLibrary(self.store.root.parent)
        self.macros = load_macros()
        self.presets = load_presets()
        self.settings = DeviceSettings.load(self.store.root.parent)
        block = self.settings.block_size or None
        self.engine = DspEngine(macros=self.macros, block_size=block)
        self.path = RoleplayPath(self.engine, self.settings)
        # Skip importing torch here — that can stall first paint on GPU boxes.
        self.neural: NeuralRuntime = probe_runtime(
            self.store.root.parent, check_torch=False
        )
        self.neural_engine: NeuralEngine | None = None
        self.neural_error = ""
        self._neural_lock = threading.Lock()
        self._neural_loading = False
        self._neural_preload_active = False
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
        if engine_id == NEURAL_ENGINE_ID:
            return self.neural.ready
        return engine_id in INSTALLED_ENGINE_IDS

    @property
    def neural_loading(self) -> bool:
        return self._neural_preload_active or self._neural_loading

    def begin_neural_preload(self) -> None:
        """Mark the launch X-VC load (includes the deferred torch probe)."""
        self._neural_preload_active = True

    def end_neural_preload(self) -> None:
        self._neural_preload_active = False

    def should_preload_neural(self) -> bool:
        """True when X-VC should load at launch (DSP-only users skip this)."""
        if self.neural_engine is not None:
            return False
        if not self.neural.conversion.installed:
            return False
        has_voice = any(voice.engine_id == NEURAL_ENGINE_ID for voice in self.voices)
        return bool(self.neural.gpu_ok or self.neural.ready or has_voice)

    def refresh_neural(self) -> None:
        """Re-probe after a pack download; drops a stale Engine instance."""
        self.neural = probe_runtime(self.store.root.parent, gpu=self.neural.gpu)
        self.neural_error = ""
        if not self.neural.ready:
            self.neural_engine = None

    def ensure_neural_engine(self, *, from_loader: bool = False) -> NeuralEngine | None:
        """Load the Neural Engine on first use (it is a multi-GB model).

        The UI thread must pass the default so a launch preload cannot freeze
        Preview. The background loader uses ``from_loader=True``.
        """
        if self.neural_engine is not None:
            return self.neural_engine
        if self.neural_loading and not from_loader:
            return None
        if not self._neural_lock.acquire(blocking=False):
            return None
        try:
            if self.neural_engine is not None:
                return self.neural_engine
            if not self.neural.ready or self.neural_error:
                return None
            self._neural_loading = True
            try:
                self.neural_engine = build_neural_engine(
                    self.neural, block_size=self.engine.block_size
                )
            except NeuralUnavailable as exc:
                self.neural_error = f"Neural Engine failed to start: {exc}"
                log.warning(self.neural_error)
            finally:
                self._neural_loading = False
            return self.neural_engine
        finally:
            self._neural_lock.release()

    def preview_engine(self):
        """Engine Preview should render the draft with (Neural Voice → Neural)."""
        if self.draft.engine_id == NEURAL_ENGINE_ID:
            engine = self.ensure_neural_engine()
            if engine is not None:
                engine.set_params(self.resolved_params(self.draft.params))
                return engine
        return self.engine

    def resolved_params(self, params: dict) -> dict:
        """Point ``reference_clip`` at the library file for ``reference_clip_id``.

        The id is what a Voice really stores; the path is derived so a moved
        data folder or a re-imported clip does not strand the Voice.
        """
        merged = dict(params)
        clip_id = str(merged.get(REFERENCE_CLIP_ID_KEY) or "")
        if clip_id:
            path = self.clips.path_for(clip_id)
            if path.is_file():
                merged[REFERENCE_CLIP_KEY] = str(path)
        return merged

    def set_engine_kind(self, engine_id: str) -> None:
        """Switch the draft between the DSP Engine and the Neural mimic Engine."""
        if engine_id not in (DSP_ENGINE_ID, NEURAL_ENGINE_ID):
            return
        self.draft.engine_id = engine_id
        if engine_id == NEURAL_ENGINE_ID:
            self.draft.params.setdefault("mix", 1.0)

    def use_clip(self, clip_id: str) -> Clip | None:
        """Make ``clip_id`` the draft's mimic reference; updates a loaded Engine."""
        clip = self.clips.get(clip_id)
        if clip is None:
            return None
        self.draft.params[REFERENCE_CLIP_ID_KEY] = clip.id
        self.draft.params[REFERENCE_CLIP_KEY] = str(self.clips.path_for(clip.id))
        if self.neural_engine is not None and self.draft.engine_id == NEURAL_ENGINE_ID:
            self.neural_engine.set_params(self.resolved_params(self.draft.params))
        return clip

    def draft_clip(self) -> Clip | None:
        return self.clips.get(str(self.draft.params.get(REFERENCE_CLIP_ID_KEY) or ""))

    def clear_clip(self) -> None:
        self.draft.params.pop(REFERENCE_CLIP_ID_KEY, None)
        self.draft.params.pop(REFERENCE_CLIP_KEY, None)

    def voice_by_id(self, voice_id: str) -> Voice | None:
        return next((voice for voice in self.voices if voice.id == voice_id), None)

    def replace_voice(self, voice: Voice) -> None:
        self.voices = [item for item in self.voices if item.id != voice.id]
        self.voices.append(voice)

    def full_params(self, params: dict | None = None) -> dict:
        merged = {spec.key: spec.default for spec in self.engine.parameter_schema()}
        merged.update(params or {})
        return merged

    def apply_draft_to_engine(self) -> None:
        self.engine.set_params(self.full_params(self.draft.params))

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

    def saved_voice_for_preset(self, name: str) -> Voice | None:
        """The GM's most recently saved Voice made from this preset, if any."""
        wanted = name.strip().lower()
        matches = [
            voice
            for voice in self.voices
            if voice.preset.lower() == wanted and self.engine_installed(voice.engine_id)
        ]
        if not matches:
            return None
        return max(matches, key=lambda voice: (voice.updated, voice.created))

    def unique_voice_name(self, name: str) -> str:
        taken = {voice.name.lower() for voice in self.voices}
        if name.lower() not in taken:
            return name
        number = 2
        while f"{name} {number}".lower() in taken:
            number += 1
        return f"{name} {number}"

    def edit_from_preset(self, name: str, *, fresh: bool = False) -> Voice | None:
        """Open a preset: the GM's saved Voice for it wins; else a new draft.

        ``fresh=True`` always starts a new unsaved draft from the bundled recipe,
        named so it does not collide with a saved Voice.
        """
        preset = preset_by_name(self.presets, name)
        if preset is None:
            return None
        if not fresh:
            saved = self.saved_voice_for_preset(preset.name)
            if saved is not None:
                self.edit(saved)
                return self.draft
        self.draft = voice_from_preset(preset)
        self.draft.name = self.unique_voice_name(preset.name)
        self._saved = Voice.new().to_dict()
        self.apply_draft_to_engine()
        return self.draft

    def design_from_hints(self, prompt: str | None = None) -> VoiceDesign:
        """Prompt → Voice: fill the draft's tags and sliders from its Tone Hints."""
        text = prompt if prompt is not None else self.draft.tone_hints
        design = design_voice(text, self.macros, self.neural)
        self.draft.tone_hints = text
        self.draft.tone_tags = list(design.tone_tags)
        self.draft.params = dict(design.params)
        self.draft.engine_id = design.engine_id
        if not self.draft.name.strip() or self.draft.name == "New Voice":
            self.draft.name = design.name
        self.apply_draft_to_engine()
        return design

    def set_active(self, voice_id: str) -> Voice | None:
        voice = self.voice_by_id(voice_id)
        if voice is None or not self.engine_installed(voice.engine_id):
            return None
        voice.touch(used=True)
        self.store.save(voice)
        self.active_id = voice.id
        fade = 30.0 if self.roleplay_on else 0.0
        self.engine.set_params(self.full_params(voice.params), crossfade_ms=fade)
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

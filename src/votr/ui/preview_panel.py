# Copyright (C) 2026 Michael Cobb
# SPDX-License-Identifier: GPL-3.0-or-later

"""Preview panel: push-to-talk Take, render through the Voice, play speakers."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from votr.neural import NEURAL_ENGINE_ID
from votr.preview import (
    MAX_TAKE_SECONDS,
    input_devices,
    keep_take,
    load_sample_take,
    play_on_speakers,
    render_take,
    save_last_take,
    speaker_devices,
    stop_playback,
)
from votr.session import Session


class PreviewPanel(QWidget):
    def __init__(self, session: Session, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.session = session
        self._recording = False
        self._compare_dry = False
        self._chunks: list = []
        self._stream = None
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(300)
        self._debounce.timeout.connect(self._auto_replay)
        self._build()
        self._refresh_status()
        if session.take is None:
            session.take = load_sample_take()

    def _build(self) -> None:
        root = QVBoxLayout(self)
        self.status = QLabel()
        self.status.setWordWrap(True)
        root.addWidget(self.status)
        self.meter = QProgressBar()
        self.meter.setRange(0, 100)
        self.countdown = QLabel("")
        root.addWidget(self.meter)
        root.addWidget(self.countdown)
        self.record = QPushButton("Hold to record")
        self.record.setCheckable(True)
        self.record.pressed.connect(self._start_record)
        self.record.released.connect(self._stop_record)
        root.addWidget(self.record)
        row = QHBoxLayout()
        play = QPushButton("Play again")
        play.clicked.connect(lambda: self.render_and_play(force=True))
        stop = QPushButton("Stop")
        stop.clicked.connect(stop_playback)
        keep = QPushButton("Keep this Take")
        keep.clicked.connect(self._keep)
        row.addWidget(play)
        row.addWidget(stop)
        row.addWidget(keep)
        root.addLayout(row)
        self.auto = QCheckBox("Auto-replay when sliders move")
        self.auto.setChecked(True)
        self.auto.toggled.connect(self._set_auto)
        self.dry = QCheckBox("Compare with dry")
        self.dry.toggled.connect(self._set_dry)
        root.addWidget(self.auto)
        root.addWidget(self.dry)

    def _refresh_status(self) -> None:
        mic = bool(input_devices())
        speakers = bool(speaker_devices())
        bits = []
        if not mic:
            bits.append("No microphone — using the bundled sample phrase.")
        if not speakers:
            bits.append("No speakers — render still runs; playback is skipped.")
        self.status.setText(" ".join(bits) or "Record a Take, then hear the Voice.")

    def _set_auto(self, checked: bool) -> None:
        self.session.auto_replay = checked

    def _set_dry(self, checked: bool) -> None:
        self._compare_dry = checked
        self.render_and_play(force=True)

    def _start_record(self) -> None:
        self._recording = True
        self._chunks = []
        self.countdown.setText(f"0 / {MAX_TAKE_SECONDS}s")
        self.meter.setValue(0)
        if not input_devices():
            return
        try:
            import sounddevice as sd

            def callback(indata, frames, time_info, status) -> None:
                self._chunks.append(indata[:, 0].copy())

            self._stream = sd.InputStream(
                samplerate=48_000,
                channels=1,
                dtype="float32",
                callback=callback,
            )
            self._stream.start()
        except Exception:
            self._stream = None

    def _stop_record(self) -> None:
        if not self._recording:
            return
        self._recording = False
        self.record.setChecked(False)
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        if self._chunks:
            take = np.concatenate(self._chunks)
            take = take[: int(MAX_TAKE_SECONDS * 48_000)]
        else:
            take = load_sample_take()
        self.session.take = take
        save_last_take(self.session.store.root.parent, take)
        self.meter.setValue(int(min(100, float(abs(take).max()) * 100)))
        self.countdown.setText("Take ready")
        self.render_and_play(force=True)

    def schedule_replay(self) -> None:
        if self.session.auto_replay:
            self._debounce.start()

    def _auto_replay(self) -> None:
        self.render_and_play(force=False)

    def render_and_play(self, *, force: bool, voice_params: dict | None = None) -> None:
        take = self.session.take
        if take is None:
            take = load_sample_take()
            self.session.take = take
        engine = self.session.engine
        if voice_params is not None:
            saved = engine.params()
            engine.set_params(self.session.full_params(voice_params))
        else:
            saved = None
            self.session.apply_draft_to_engine()
            engine = self.session.preview_engine()
        rendered = take.copy() if self._compare_dry else render_take(engine, take)
        if saved is not None:
            engine.set_params(saved)
        played = play_on_speakers(rendered)
        if not played and force:
            self.status.setText(
                self.status.text() + " Rendered without playback (no speakers)."
            )

    def preview_other_voice(self, voice_id: str) -> None:
        voice = self.session.voice_by_id(voice_id)
        if voice is None:
            return
        if voice.engine_id == NEURAL_ENGINE_ID:
            engine = self.session.ensure_neural_engine()
            if engine is None:
                self.status.setText("Neural Voice — the Neural Engine is not ready.")
                return
            engine.set_params(voice.params)
            take = (
                self.session.take
                if self.session.take is not None
                else load_sample_take()
            )
            play_on_speakers(render_take(engine, take))
            return
        self.render_and_play(force=True, voice_params=voice.params)

    def _keep(self) -> None:
        if self.session.take is None:
            return
        name, ok = QInputDialog.getText(self, "Keep this Take", "Name")
        if not ok or not name.strip():
            return
        keep_take(self.session.store.root.parent, self.session.take, name)

    def eventFilter(self, obj, event) -> bool:  # noqa: N802
        if event.type() == event.Type.KeyPress and event.key() == Qt.Key.Key_Space:
            if not event.isAutoRepeat():
                self._start_record()
            return True
        if event.type() == event.Type.KeyRelease and event.key() == Qt.Key.Key_Space:
            if not event.isAutoRepeat():
                self._stop_record()
            return True
        return super().eventFilter(obj, event)

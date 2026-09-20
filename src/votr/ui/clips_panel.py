# Copyright (C) 2026 Michael Cobb
# SPDX-License-Identifier: GPL-3.0-or-later

"""Mimic clip panel: pick, upload, record, rename and delete reference clips."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import QTimer, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from votr.clips import (
    CLIP_SAMPLE_RATE,
    IMPORT_EXTENSIONS,
    MAX_CLIP_SECONDS,
    ClipError,
)
from votr.preview import input_devices
from votr.session import Session


class MimicClipPanel(QGroupBox):
    """Shown for Neural Voices. Emits ``clip_chosen`` when the draft's clip changes."""

    clip_chosen = Signal(str)

    def __init__(self, session: Session, parent: QWidget | None = None) -> None:
        super().__init__("Mimic clip — the voice to sound like", parent)
        self.session = session
        self._recording = False
        self._chunks: list[np.ndarray] = []
        self._stream = None
        self._elapsed = 0.0
        self._tick = QTimer(self)
        self._tick.setInterval(100)
        self._tick.timeout.connect(self._on_tick)
        self._build()
        self.refresh()

    def _build(self) -> None:
        root = QVBoxLayout(self)
        hint = QLabel(
            "5–10 seconds of clear speech from the voice you want. Record it, or "
            "upload a WAV/FLAC/MP3. Clips live in your data folder; one clip can "
            "serve many Voices."
        )
        hint.setWordWrap(True)
        root.addWidget(hint)

        pick = QHBoxLayout()
        self.clip_box = QComboBox()
        self.clip_box.setObjectName("clip_box")
        self.use_button = QPushButton("Use this clip")
        self.use_button.clicked.connect(self.use_selected)
        pick.addWidget(self.clip_box, 1)
        pick.addWidget(self.use_button)
        root.addLayout(pick)

        actions = QHBoxLayout()
        upload = QPushButton("Upload…")
        upload.clicked.connect(self._upload_dialog)
        self.record_button = QPushButton(f"Record ({MAX_CLIP_SECONDS:.0f} s max)")
        self.record_button.setObjectName("clip_record")
        self.record_button.setCheckable(True)
        self.record_button.clicked.connect(self._toggle_record)
        rename = QPushButton("Rename…")
        rename.clicked.connect(self._rename_dialog)
        delete = QPushButton("Delete")
        delete.clicked.connect(self._delete_dialog)
        for button in (upload, self.record_button, rename, delete):
            actions.addWidget(button)
        root.addLayout(actions)

        self.progress = QProgressBar()
        self.progress.setRange(0, int(MAX_CLIP_SECONDS * 10))
        self.progress.setVisible(False)
        root.addWidget(self.progress)
        self.status = QLabel()
        self.status.setObjectName("clip_status")
        self.status.setWordWrap(True)
        root.addWidget(self.status)

    # -- state ------------------------------------------------------------

    def selected_id(self) -> str:
        return str(self.clip_box.currentData() or "")

    def refresh(self, select_id: str | None = None) -> None:
        """Rebuild the dropdown, keeping the selection (or ``select_id``) put."""
        current = self.session.draft_clip()
        wanted = select_id or self.selected_id() or (current.id if current else "")
        self.clip_box.blockSignals(True)
        self.clip_box.clear()
        clips = self.session.clips.list()
        if not clips:
            self.clip_box.addItem("No clips yet — record or upload one", "")
        for clip in clips:
            self.clip_box.addItem(clip.label, clip.id)
        index = self.clip_box.findData(wanted) if wanted else -1
        if index < 0 and current is not None:
            index = self.clip_box.findData(current.id)
        if index >= 0:
            self.clip_box.setCurrentIndex(index)
        self.clip_box.blockSignals(False)
        self.use_button.setEnabled(bool(clips))
        if current is not None:
            self.status.setText(f"This Voice mimics “{current.name}”.")
        elif clips:
            self.status.setText("Pick a clip and press “Use this clip”.")
        else:
            self.status.setText("This Voice has no mimic clip yet.")

    # -- select -----------------------------------------------------------

    def use_selected(self) -> None:
        clip_id = self.selected_id()
        if not clip_id:
            return
        clip = self.session.use_clip(clip_id)
        if clip is None:
            self.status.setText("That clip is gone; pick another.")
            self.refresh()
            return
        self.refresh()
        self.clip_chosen.emit(clip.id)

    # -- upload -----------------------------------------------------------

    def add_file(self, path: Path, name: str | None = None) -> bool:
        try:
            clip = self.session.clips.import_file(Path(path), name)
        except ClipError as exc:
            self.status.setText(str(exc))
            return False
        self._select_and_use(clip.id)
        self.status.setText(f"Added “{clip.name}” ({clip.seconds:.1f} s) and using it.")
        return True

    def _upload_dialog(self) -> None:
        patterns = " ".join(f"*{ext}" for ext in IMPORT_EXTENSIONS)
        path, _ = QFileDialog.getOpenFileName(
            self, "Upload a mimic clip", "", f"Audio files ({patterns})"
        )
        if path:
            self.add_file(Path(path))

    # -- record -----------------------------------------------------------

    def _toggle_record(self, checked: bool) -> None:
        if checked:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self) -> bool:
        if self._recording:
            return True
        if not input_devices():
            self.status.setText("No microphone found — upload a clip instead.")
            self.record_button.setChecked(False)
            return False
        self._chunks = []
        self._elapsed = 0.0
        try:
            import sounddevice as sd

            def callback(indata, frames, time_info, status) -> None:
                self._chunks.append(indata[:, 0].copy())

            self._stream = sd.InputStream(
                samplerate=CLIP_SAMPLE_RATE,
                channels=1,
                dtype="float32",
                callback=callback,
            )
            self._stream.start()
        except Exception as exc:  # noqa: BLE001 — device errors are user-facing
            self._stream = None
            self.status.setText(f"Could not open the microphone: {exc}")
            self.record_button.setChecked(False)
            return False
        self._recording = True
        self.record_button.setChecked(True)
        self.record_button.setText("Stop")
        self.progress.setValue(0)
        self.progress.setVisible(True)
        self._tick.start()
        self._show_remaining()
        return True

    def _show_remaining(self) -> None:
        remaining = max(0.0, MAX_CLIP_SECONDS - self._elapsed)
        self.progress.setValue(int(self._elapsed * 10))
        self.status.setText(
            f"Recording… {remaining:.0f} s left. Speak as the character."
        )

    def _on_tick(self) -> None:
        self._elapsed += self._tick.interval() / 1000.0
        if self._elapsed >= MAX_CLIP_SECONDS:
            self.stop_recording()
            return
        self._show_remaining()

    def stop_recording(self, name: str | None = None) -> bool:
        """Stop and save. ``name=None`` asks; tests pass a name."""
        if not self._recording:
            return False
        self._recording = False
        self._tick.stop()
        self.record_button.setChecked(False)
        self.record_button.setText(f"Record ({MAX_CLIP_SECONDS:.0f} s max)")
        self.progress.setVisible(False)
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:  # noqa: BLE001
                pass
            self._stream = None
        audio = (
            np.concatenate(self._chunks) if self._chunks else np.zeros(0, np.float32)
        )
        return self.save_recording(audio, name)

    def save_recording(self, audio: np.ndarray, name: str | None = None) -> bool:
        if name is None:
            name, ok = QInputDialog.getText(self, "Name this mimic clip", "Name")
            if not ok:
                self.status.setText("Recording discarded.")
                return False
        try:
            clip = self.session.clips.add(audio, CLIP_SAMPLE_RATE, name or "Mimic clip")
        except ClipError as exc:
            self.status.setText(str(exc))
            return False
        self._select_and_use(clip.id)
        self.status.setText(f"Saved “{clip.name}” ({clip.seconds:.1f} s) and using it.")
        return True

    def _select_and_use(self, clip_id: str) -> None:
        self.session.use_clip(clip_id)
        self.refresh(select_id=clip_id)
        self.clip_chosen.emit(clip_id)

    # -- rename / delete --------------------------------------------------

    def rename_selected(self, name: str) -> bool:
        clip_id = self.selected_id()
        if not clip_id:
            return False
        clip = self.session.clips.rename(clip_id, name)
        self.refresh()
        return clip is not None

    def _rename_dialog(self) -> None:
        clip_id = self.selected_id()
        clip = self.session.clips.get(clip_id) if clip_id else None
        if clip is None:
            return
        name, ok = QInputDialog.getText(self, "Rename clip", "Name", text=clip.name)
        if ok and name.strip():
            self.rename_selected(name)

    def delete_selected(self) -> bool:
        clip_id = self.selected_id()
        if not clip_id:
            return False
        if (
            self.session.draft_clip() is not None
            and self.session.draft_clip().id == clip_id
        ):
            self.session.clear_clip()
        self.session.clips.delete(clip_id)
        self.clip_box.setCurrentIndex(-1)
        self.refresh()
        self.status.setText("Clip deleted. Voices that used it need a new clip.")
        return True

    def _delete_dialog(self) -> None:
        clip_id = self.selected_id()
        clip = self.session.clips.get(clip_id) if clip_id else None
        if clip is None:
            return
        answer = QMessageBox.question(self, "Delete clip", f"Delete “{clip.name}”?")
        if answer == QMessageBox.StandardButton.Yes:
            self.delete_selected()

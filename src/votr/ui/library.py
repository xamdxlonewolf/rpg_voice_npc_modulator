# Copyright (C) 2026 Michael Cobb
# SPDX-License-Identifier: GPL-3.0-or-later

"""Voice Library grid of Voice Buttons."""

from __future__ import annotations

from PySide6.QtCore import QPoint, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QAction, QDesktopServices
from PySide6.QtWidgets import (
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from votr.session import Session
from votr.voice import Voice


class VoiceButton(QPushButton):
    edit_requested = Signal(str)
    duplicate_requested = Signal(str)
    delete_requested = Signal(str)
    reveal_requested = Signal(str)
    preview_requested = Signal(str)

    def __init__(self, voice: Voice, *, active: bool, enabled: bool) -> None:
        tags = ", ".join(voice.tone_tags[:2])
        label = voice.name if not tags else f"{voice.name}\n{tags}"
        super().__init__(label)
        self.voice_id = voice.id
        self.setMinimumSize(140, 88)
        self.setCheckable(True)
        self.setChecked(active)
        self.setEnabled(enabled)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet(
            f"background: {voice.colour}; text-align: center; padding: 8px;"
        )
        self.setToolTip(voice.tone_hints or voice.name)
        if not enabled:
            self.setToolTip(
                f"Engine “{voice.engine_id}” is not installed. This Voice is disabled."
            )
        self._press_timer = QTimer(self)
        self._press_timer.setSingleShot(True)
        self._press_timer.timeout.connect(self._long_press)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._menu)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self._press_timer.start(550)
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        self._press_timer.stop()
        super().mouseReleaseEvent(event)

    def _long_press(self) -> None:
        self._menu(self.rect().center())

    def _menu(self, pos: QPoint) -> None:
        menu = QMenu(self)
        edit = QAction("Edit", self)
        edit.triggered.connect(lambda: self.edit_requested.emit(self.voice_id))
        dup = QAction("Duplicate", self)
        dup.triggered.connect(lambda: self.duplicate_requested.emit(self.voice_id))
        delete = QAction("Delete", self)
        delete.triggered.connect(lambda: self.delete_requested.emit(self.voice_id))
        reveal = QAction("Reveal file", self)
        reveal.triggered.connect(lambda: self.reveal_requested.emit(self.voice_id))
        preview = QAction("Preview Take", self)
        preview.triggered.connect(lambda: self.preview_requested.emit(self.voice_id))
        for action in (edit, dup, delete, reveal, preview):
            menu.addAction(action)
        menu.exec(self.mapToGlobal(self.rect().center()))


class VoiceLibrary(QWidget):
    edit_voice = Signal(str)
    new_voice = Signal()
    preview_voice = Signal(str)

    def __init__(self, session: Session, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.session = session
        root = QVBoxLayout(self)
        tools = QHBoxLayout()
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search name or tag")
        self.search.textChanged.connect(self.refresh)
        self.sort = QComboBox()
        self.sort.addItems(["Name", "Last used"])
        self.sort.currentIndexChanged.connect(self.refresh)
        tools.addWidget(self.search)
        tools.addWidget(self.sort)
        root.addLayout(tools)
        self.empty = QLabel("Create your first Voice — click New Voice.")
        self.empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(self.empty)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self._grid_host = QWidget()
        self.grid = QGridLayout(self._grid_host)
        scroll.setWidget(self._grid_host)
        root.addWidget(scroll)
        self.refresh()

    def refresh(self) -> None:
        while self.grid.count():
            item = self.grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        voices = self._filtered()
        self.empty.setVisible(not self.session.voices)
        columns = 3
        self._add_new_tile(0)
        for index, voice in enumerate(voices, start=1):
            button = VoiceButton(
                voice,
                active=voice.id == self.session.active_id,
                enabled=self.session.engine_installed(voice.engine_id),
            )
            button.clicked.connect(lambda _=False, vid=voice.id: self._select(vid))
            button.edit_requested.connect(self.edit_voice)
            button.duplicate_requested.connect(self._duplicate)
            button.delete_requested.connect(self._delete)
            button.reveal_requested.connect(self._reveal)
            button.preview_requested.connect(self.preview_voice)
            self.grid.addWidget(button, index // columns, index % columns)

    def _add_new_tile(self, index: int) -> None:
        tile = QPushButton("New Voice")
        tile.setMinimumSize(140, 88)
        tile.clicked.connect(self.new_voice)
        self.grid.addWidget(tile, index // 3, index % 3)

    def _filtered(self) -> list[Voice]:
        query = self.search.text().strip().lower()
        voices = list(self.session.voices)
        if query:
            voices = [
                voice
                for voice in voices
                if query in voice.name.lower()
                or any(query in tag.lower() for tag in voice.tone_tags)
            ]
        if self.sort.currentText() == "Last used":
            voices.sort(key=lambda voice: voice.last_used or "", reverse=True)
        else:
            voices.sort(key=lambda voice: voice.name.lower())
        return voices

    def _select(self, voice_id: str) -> None:
        self.session.set_active(voice_id)
        self.refresh()
        parent = self.window()
        if hasattr(parent, "refresh_chrome"):
            parent.refresh_chrome()

    def _duplicate(self, voice_id: str) -> None:
        voice = self.session.voice_by_id(voice_id)
        if voice is None:
            return
        self.session.duplicate_voice(voice)
        self.refresh()

    def _delete(self, voice_id: str) -> None:
        voice = self.session.voice_by_id(voice_id)
        if voice is None:
            return
        answer = QMessageBox.question(
            self, "Delete Voice", f"Delete “{voice.name}”?"
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self.session.delete_voice(voice_id)
        self.refresh()
        parent = self.window()
        if hasattr(parent, "refresh_chrome"):
            parent.refresh_chrome()

    def _reveal(self, voice_id: str) -> None:
        path = self.session.store.path_for(voice_id)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.parent)))

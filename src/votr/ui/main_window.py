# Copyright (C) 2026 Michael Cobb
# SPDX-License-Identifier: GPL-3.0-or-later

"""Main window: Voice Library, editor, Roleplay chrome."""

from __future__ import annotations

from PySide6.QtWidgets import QLabel, QMainWindow, QStackedWidget, QVBoxLayout, QWidget

from votr import WINDOW_TITLE
from votr.session import Session
from votr.ui.editor import VoiceEditor
from votr.ui.library import VoiceLibrary
from votr.ui.roleplay_panel import RoleplayPanel


class MainWindow(QMainWindow):
    def __init__(self, session: Session) -> None:
        super().__init__()
        self.session = session
        self.setWindowTitle(WINDOW_TITLE)
        self.resize(880, 820)
        central = QWidget()
        layout = QVBoxLayout(central)
        self.header = QLabel()
        layout.addWidget(self.header)
        self.pages = QStackedWidget()
        self.library = VoiceLibrary(session)
        self.editor = VoiceEditor(session)
        self.pages.addWidget(self.library)
        self.pages.addWidget(self.editor)
        layout.addWidget(self.pages, 1)
        self.roleplay = RoleplayPanel(session)
        layout.addWidget(self.roleplay)
        self.setCentralWidget(central)
        self.library.edit_voice.connect(self.show_editor)
        self.library.new_voice.connect(self.show_new_editor)
        self.library.preview_voice.connect(self.editor.preview.preview_other_voice)
        self.editor.voice_saved.connect(self._after_save)
        self.editor.voice_deleted.connect(self.show_library)
        self.refresh_chrome()

    def refresh_chrome(self) -> None:
        active = self.session.active_voice()
        name = active.name if active is not None else "none"
        self.header.setText(f"Active Voice: {name}")
        self.roleplay.refresh()

    def show_library(self) -> None:
        self.library.refresh()
        self.pages.setCurrentWidget(self.library)
        self.refresh_chrome()

    def show_editor(self, voice_id: str) -> None:
        if not self.editor.confirm_discard():
            return
        voice = self.session.voice_by_id(voice_id)
        if voice is None:
            return
        self.session.edit(voice)
        self.editor.reload_from_draft()
        self.pages.setCurrentWidget(self.editor)

    def show_new_editor(self) -> None:
        if not self.editor.confirm_discard():
            return
        self.session.edit_new()
        self.editor.reload_from_draft()
        self.pages.setCurrentWidget(self.editor)

    def _after_save(self) -> None:
        self.show_library()

    def closeEvent(self, event) -> None:  # noqa: N802 — Qt
        if self.editor.confirm_discard():
            event.accept()
        else:
            event.ignore()

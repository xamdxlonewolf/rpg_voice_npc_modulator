# Copyright (C) 2026 Michael Cobb
# SPDX-License-Identifier: GPL-3.0-or-later

"""Main window: header + Voice editor (library arrives in E2)."""

from __future__ import annotations

from PySide6.QtWidgets import QLabel, QMainWindow, QVBoxLayout, QWidget

from votr import WINDOW_TITLE
from votr.session import Session
from votr.ui.editor import VoiceEditor


class MainWindow(QMainWindow):
    def __init__(self, session: Session) -> None:
        super().__init__()
        self.session = session
        self.setWindowTitle(WINDOW_TITLE)
        self.resize(720, 780)
        central = QWidget()
        layout = QVBoxLayout(central)
        self.header = QLabel()
        layout.addWidget(self.header)
        self.editor = VoiceEditor(session)
        self.editor.voice_saved.connect(self._refresh_header)
        self.editor.voice_deleted.connect(self._refresh_header)
        layout.addWidget(self.editor)
        self.setCentralWidget(central)
        self._refresh_header()

    def _refresh_header(self) -> None:
        active = self.session.active_voice()
        name = active.name if active is not None else "none"
        self.header.setText(f"Active Voice: {name}")

    def closeEvent(self, event) -> None:  # noqa: N802 — Qt
        if self.editor.confirm_discard():
            event.accept()
        else:
            event.ignore()

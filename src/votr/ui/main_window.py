# Copyright (C) 2026 Michael Cobb
# SPDX-License-Identifier: GPL-3.0-or-later

"""Main window: Voice Library, editor, Roleplay chrome."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QKeySequence, QShortcut
from PySide6.QtWidgets import QLabel, QMainWindow, QStackedWidget, QVBoxLayout, QWidget

from votr import WINDOW_TITLE
from votr.session import Session
from votr.ui.editor import VoiceEditor
from votr.ui.first_run import FirstRunWizard
from votr.ui.library import VoiceLibrary
from votr.ui.neural_preload import NeuralLoadOverlay, NeuralLoadWorker
from votr.ui.roleplay_panel import RoleplayPanel
from votr.ui.settings import SettingsDialog
from votr.ui.wizard import DiscordWizard


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
        self.roleplay.wizard_requested.connect(self.open_wizard)
        settings_menu = self.menuBar().addMenu("Settings")
        open_settings = QAction("Settings…", self)
        open_settings.triggered.connect(self.open_settings)
        first_run = QAction("First-run setup…", self)
        first_run.triggered.connect(self.open_first_run)
        discord = QAction("Discord setup…", self)
        discord.triggered.connect(self.open_wizard)
        settings_menu.addAction(open_settings)
        settings_menu.addAction(first_run)
        settings_menu.addAction(discord)
        self.panic_shortcut = QShortcut(
            QKeySequence(session.settings.panic_hotkey), self
        )
        self.panic_shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)
        self.panic_shortcut.activated.connect(self.roleplay.toggle_panic)
        self._neural_overlay: NeuralLoadOverlay | None = None
        self._neural_worker: NeuralLoadWorker | None = None
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

    def open_wizard(self) -> None:
        wizard = DiscordWizard(self.session, self)
        wizard.exec()
        self.roleplay.refresh()

    def open_first_run(self) -> None:
        wizard = FirstRunWizard(self.session, self)
        wizard.exec()
        self.roleplay.refresh()

    def open_settings(self) -> None:
        dialog = SettingsDialog(self.session, self)
        dialog.exec()
        self.panic_shortcut.setKey(QKeySequence(self.session.settings.panic_hotkey))
        self.roleplay.refresh()

    def start_neural_preload(self) -> NeuralLoadWorker | None:
        """Show the spinner and load X-VC after first paint. DSP-only skips."""
        if self._neural_worker is not None and self._neural_worker.isRunning():
            return self._neural_worker
        if not self.session.should_preload_neural():
            return None
        host = self.centralWidget() or self
        overlay = NeuralLoadOverlay(host)
        overlay.setGeometry(host.rect())
        overlay.show()
        overlay.raise_()
        worker = NeuralLoadWorker(self.session, self)
        worker.finished_ok.connect(self._on_neural_preload_done)
        worker.failed.connect(self._on_neural_preload_failed)
        self._neural_overlay = overlay
        self._neural_worker = worker
        worker.start()
        return worker

    def _place_neural_overlay(self) -> None:
        overlay = self._neural_overlay
        if overlay is None:
            return
        host = overlay.parentWidget()
        overlay.setGeometry(host.rect() if host is not None else self.rect())

    def resizeEvent(self, event) -> None:  # noqa: N802 — Qt
        super().resizeEvent(event)
        self._place_neural_overlay()

    def _clear_neural_overlay(self) -> None:
        overlay = self._neural_overlay
        self._neural_overlay = None
        if overlay is not None:
            overlay.hide()
            overlay.deleteLater()
        self.library.refresh()
        self.editor.reload_from_draft()
        self.refresh_chrome()

    def _on_neural_preload_done(self) -> None:
        self._clear_neural_overlay()

    def _on_neural_preload_failed(self, _message: str) -> None:
        self._clear_neural_overlay()

    def closeEvent(self, event) -> None:  # noqa: N802 — Qt
        if self.editor.confirm_discard():
            self.session.stop_roleplay()
            event.accept()
        else:
            event.ignore()

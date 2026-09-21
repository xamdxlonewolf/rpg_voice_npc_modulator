# Copyright (C) 2026 Michael Cobb
# SPDX-License-Identifier: GPL-3.0-or-later

"""Voice of the Realm application entry."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from votr.session import Session

HEADLESS_ENV = "VOTR_HEADLESS"


def create_application(argv: list[str] | None = None):
    """Return the process QApplication, creating one if needed."""
    from PySide6.QtWidgets import QApplication

    existing = QApplication.instance()
    if existing is not None:
        return existing
    return QApplication(list(sys.argv if argv is None else argv))


def create_main_window(session: Session | None = None, data_dir: Path | None = None):
    """Build the main window with the Voice editor."""
    from votr.ui.main_window import MainWindow

    if session is None:
        session = Session(data_dir)
    return MainWindow(session)


def is_headless(argv: list[str]) -> bool:
    return "--headless" in argv or os.environ.get(HEADLESS_ENV) == "1"


def run(argv: list[str] | None = None) -> int:
    """Show the window, or construct it and exit when headless."""
    args = list(sys.argv[1:] if argv is None else argv)
    if is_headless(args):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = create_application(["votr", *args])
    window = create_main_window()
    if is_headless(args):
        return 0
    window.show()
    # First paint before any X-VC / torch work so Windows never sits unpainted.
    app.processEvents()
    window.start_neural_preload()
    if not window.session.settings.first_run_completed:
        window.open_first_run()
    return app.exec()

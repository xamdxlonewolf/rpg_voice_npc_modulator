# Copyright (C) 2026 Michael Cobb
# SPDX-License-Identifier: GPL-3.0-or-later

"""Empty Voice of the Realm window (S0.1 scaffold)."""

from __future__ import annotations

import os
import sys

from votr import WINDOW_TITLE

HEADLESS_ENV = "VOTR_HEADLESS"


def create_application(argv: list[str] | None = None):
    """Return the process QApplication, creating one if needed."""
    from PySide6.QtWidgets import QApplication

    existing = QApplication.instance()
    if existing is not None:
        return existing
    return QApplication(list(sys.argv if argv is None else argv))


def create_main_window():
    """Build the empty main window titled Voice of the Realm."""
    from PySide6.QtWidgets import QMainWindow, QWidget

    window = QMainWindow()
    window.setWindowTitle(WINDOW_TITLE)
    window.setCentralWidget(QWidget())
    return window


def is_headless(argv: list[str]) -> bool:
    return "--headless" in argv or os.environ.get(HEADLESS_ENV) == "1"


def run(argv: list[str] | None = None) -> int:
    """Show the empty window, or construct it and exit when headless."""
    args = list(sys.argv[1:] if argv is None else argv)
    if is_headless(args):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = create_application(["votr", *args])
    window = create_main_window()
    if is_headless(args):
        return 0
    window.show()
    return app.exec()

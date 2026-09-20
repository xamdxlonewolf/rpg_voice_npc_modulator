# Copyright (C) 2026 Michael Cobb
# SPDX-License-Identifier: GPL-3.0-or-later

"""Readable light theme for QWizard on Windows 11 (no white-on-gray)."""

from __future__ import annotations

from PySide6.QtWidgets import QWizard

WIZARD_STYLE = """
QWizard {
    background-color: #f4f4f4;
    color: #1a1a1a;
}
QWizard QWidget {
    color: #1a1a1a;
    background-color: #f4f4f4;
}
QWizard QLabel {
    color: #1a1a1a;
    background-color: transparent;
}
QWizard QGroupBox {
    color: #1a1a1a;
    border: 1px solid #8a8a8a;
    margin-top: 10px;
    padding-top: 8px;
}
QWizard QPushButton {
    color: #1a1a1a;
    background-color: #ffffff;
    border: 1px solid #5c5c5c;
    padding: 6px 16px;
    min-width: 88px;
}
QWizard QPushButton:hover {
    background-color: #e8e8e8;
}
QWizard QPushButton:disabled {
    color: #6a6a6a;
    background-color: #dedede;
    border: 1px solid #b0b0b0;
}
QWizard QPushButton:default {
    color: #ffffff;
    background-color: #0f4c81;
    border: 1px solid #0a365c;
}
QWizard QDialogButtonBox QPushButton {
    color: #1a1a1a;
    background-color: #ffffff;
    border: 1px solid #5c5c5c;
}
QWizard QDialogButtonBox QPushButton:default {
    color: #ffffff;
    background-color: #0f4c81;
    border: 1px solid #0a365c;
}
QWizard QComboBox, QWizard QLineEdit, QWizard QTextEdit {
    color: #1a1a1a;
    background-color: #ffffff;
    border: 1px solid #7a7a7a;
    padding: 4px;
}
QWizard QProgressBar {
    color: #1a1a1a;
    background-color: #ffffff;
    border: 1px solid #7a7a7a;
}
"""


def apply_wizard_theme(wizard: QWizard, *, title: str) -> None:
    wizard.setWizardStyle(QWizard.WizardStyle.ClassicStyle)
    wizard.setWindowTitle(title)
    wizard.setStyleSheet(WIZARD_STYLE)
    wizard.setMinimumSize(580, 460)

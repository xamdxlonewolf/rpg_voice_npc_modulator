# Copyright (C) 2026 Michael Cobb
# SPDX-License-Identifier: GPL-3.0-or-later

"""Launch overlay: load X-VC off the UI thread so Windows stays responsive."""

from __future__ import annotations

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import QLabel, QProgressBar, QVBoxLayout, QWidget

from votr.session import Session

LOADING_TEXT = "Loading Neural Engine…"


class NeuralLoadOverlay(QWidget):
    """Spinner overlay so the window keeps painting during the X-VC load."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("neural_load_overlay")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet(
            "#neural_load_overlay { background: rgba(18, 18, 18, 220); }"
            "#neural_load_overlay QLabel { color: #f4f4f4; font-size: 16px; }"
            "#neural_load_overlay QProgressBar { min-height: 18px; }"
        )
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.spinner = QProgressBar()
        self.spinner.setObjectName("neural_load_spinner")
        self.spinner.setRange(0, 0)
        self.spinner.setMaximumWidth(320)
        self.spinner.setTextVisible(False)
        self.label = QLabel(LOADING_TEXT)
        self.label.setObjectName("neural_load_label")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.spinner)
        layout.addWidget(self.label)


class NeuralLoadWorker(QThread):
    """Refresh the Neural probe and construct X-VC away from the UI thread."""

    finished_ok = Signal()
    failed = Signal(str)

    def __init__(self, session: Session, parent=None) -> None:
        super().__init__(parent)
        self._session = session

    def run(self) -> None:  # noqa: D401 — QThread entry
        session = self._session
        session.begin_neural_preload()
        try:
            session.refresh_neural()
            engine = session.ensure_neural_engine(from_loader=True)
            if engine is not None:
                self.finished_ok.emit()
            else:
                self.failed.emit(
                    session.neural_error
                    or session.neural.reason
                    or "Neural Engine failed to start"
                )
        except Exception as exc:  # pragma: no cover - defensive
            self.failed.emit(f"{type(exc).__name__}: {exc}")
        finally:
            session.end_neural_preload()

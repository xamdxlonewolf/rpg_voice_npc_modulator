# Copyright (C) 2026 Michael Cobb
# SPDX-License-Identifier: GPL-3.0-or-later

"""Mic gain slider: persistable capture gain, 0 dB = unity."""

from __future__ import annotations

import math

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QSlider, QVBoxLayout, QWidget

from votr.gain import (
    CLIPPING_PEAK,
    db_to_slider,
    format_mic_gain,
    gain_slider_steps,
    slider_to_db,
)
from votr.session import Session


class MicGainSlider(QWidget):
    """Horizontal dB slider bound to ``session.settings.mic_gain_db``."""

    changed = Signal(float)

    def __init__(self, session: Session, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.session = session
        self.setObjectName("mic_gain_slider")
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        caption = QLabel("Mic gain")
        caption.setObjectName("mic_gain_caption")
        self.value = QLabel()
        self.value.setObjectName("mic_gain_value")
        row.addWidget(caption)
        row.addWidget(self.value, 1)
        root.addLayout(row)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setObjectName("mic_gain")
        self.slider.setRange(0, gain_slider_steps())
        self.slider.setToolTip(
            "Gain on the captured microphone, before Preview or Roleplay. "
            "0 dB is unchanged — there is no extra silent boost."
        )
        self.slider.valueChanged.connect(self._moved)
        root.addWidget(self.slider)
        self.level = QLabel("Level: —")
        self.level.setObjectName("mic_gain_level")
        root.addWidget(self.level)
        self.sync_from_settings()

    def sync_from_settings(self) -> None:
        if self.slider.isSliderDown():
            return
        pos = db_to_slider(self.session.settings.mic_gain_db)
        if self.slider.value() != pos:
            self.slider.blockSignals(True)
            self.slider.setValue(pos)
            self.slider.blockSignals(False)
        self._set_value_label(self.session.settings.mic_gain_db)

    def set_peak(self, peak: float) -> None:
        """Show post-gain peak so quiet vs clipping is visible."""
        peak = max(0.0, float(peak))
        db = 20.0 * math.log10(max(peak, 1e-6))
        if peak <= 1e-6:
            self.level.setText("Level: silence")
            self.level.setStyleSheet("")
            return
        text = f"Level: {db:.0f} dBFS"
        if peak >= CLIPPING_PEAK:
            text += " — clipping, turn gain down"
            self.level.setStyleSheet("color: #b00020;")
        elif peak < 0.05:
            text += " — quiet, turn gain up"
            self.level.setStyleSheet("")
        else:
            self.level.setStyleSheet("")
        self.level.setText(text)

    def _set_value_label(self, db: float) -> None:
        self.value.setText(format_mic_gain(db))

    def _moved(self, pos: int) -> None:
        db = slider_to_db(pos)
        self._set_value_label(db)
        self.session.settings.mic_gain_db = db
        self.session.path.set_mic_gain_db(db)
        self.session.save_settings()
        self.changed.emit(db)

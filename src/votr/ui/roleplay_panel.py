# Copyright (C) 2026 Michael Cobb
# SPDX-License-Identifier: GPL-3.0-or-later

"""Roleplay Mode panel: live toggle, meters, panic, Monitor, wizard."""

from __future__ import annotations

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
)

from votr.devices import monitor_looks_like_speakers
from votr.live import AudioDeviceError
from votr.roleplay import RoleplayError
from votr.session import Session

_RECEIVING_PEAK = 0.02


class RoleplayPanel(QGroupBox):
    wizard_requested = Signal()

    def __init__(self, session: Session) -> None:
        super().__init__("Roleplay Mode")
        self.session = session
        self.setObjectName("roleplay_panel")
        layout = QVBoxLayout(self)
        self.active_label = QLabel()
        layout.addWidget(self.active_label)
        self.toggle = QPushButton("Roleplay Mode Off")
        self.toggle.setCheckable(True)
        self.toggle.setMinimumHeight(36)
        self.toggle.clicked.connect(self._toggle)
        layout.addWidget(self.toggle)
        self.state_label = QLabel("Off — only the Character Voice will reach Discord.")
        self.state_label.setWordWrap(True)
        layout.addWidget(self.state_label)
        self.reason = QLabel()
        self.reason.setWordWrap(True)
        layout.addWidget(self.reason)
        layout.addWidget(QLabel("Mic"))
        self.in_meter = QProgressBar()
        self.in_meter.setRange(0, 100)
        layout.addWidget(self.in_meter)
        layout.addWidget(QLabel("Virtual Cable"))
        self.out_meter = QProgressBar()
        self.out_meter.setRange(0, 100)
        layout.addWidget(self.out_meter)
        self.discord_hint = QLabel("")
        layout.addWidget(self.discord_hint)
        self.xruns = QLabel("Underruns 0 · Overruns 0")
        layout.addWidget(self.xruns)
        panic_row = QHBoxLayout()
        self.panic = QPushButton("Panic mute (Ctrl+Shift+M)")
        self.panic.clicked.connect(self.toggle_panic)
        panic_row.addWidget(self.panic)
        layout.addLayout(panic_row)
        self.hold = QCheckBox("Hold to talk")
        self.hold.setChecked(session.settings.hold_to_talk)
        self.hold.toggled.connect(self._hold_mode)
        layout.addWidget(self.hold)
        self.hold_btn = QPushButton("Hold")
        self.hold_btn.setCheckable(True)
        self.hold_btn.pressed.connect(lambda: session.path.set_talk_held(True))
        self.hold_btn.released.connect(lambda: session.path.set_talk_held(False))
        self.hold_btn.setVisible(session.settings.hold_to_talk)
        layout.addWidget(self.hold_btn)
        self.monitor = QCheckBox("Monitor Character Voice")
        self.monitor.setChecked(session.settings.monitor_on)
        self.monitor.toggled.connect(self._monitor)
        layout.addWidget(self.monitor)
        self.monitor_warn = QLabel(
            "Monitor is speakers — the Character Voice may feed back into the mic."
        )
        self.monitor_warn.setWordWrap(True)
        layout.addWidget(self.monitor_warn)
        self.wizard_btn = QPushButton("Discord setup…")
        self.wizard_btn.clicked.connect(self.wizard_requested)
        layout.addWidget(self.wizard_btn)
        self._timer = QTimer(self)
        self._timer.setInterval(50)
        self._timer.timeout.connect(self._tick)
        self._timer.start()
        self.refresh()

    def refresh(self) -> None:
        active = self.session.active_voice()
        name = active.name if active is not None else "none"
        self.active_label.setText(f"Active Voice: {name}")
        running = self.session.roleplay_on and self.session.path.is_running()
        self.toggle.setChecked(self.session.roleplay_on)
        if self.session.roleplay_on:
            on_label = "Roleplay Mode On"
        else:
            on_label = "Roleplay Mode Off"
        self.toggle.setText(on_label)
        if self.session.roleplay_on and running:
            self.state_label.setText("On — Character Voice only.")
        elif self.session.roleplay_on:
            self.state_label.setText("On — waiting for the audio path.")
        else:
            self.state_label.setText("Off — Character Voice only.")
        muted = self.session.path.is_muted()
        panic = "Muted — press to unmute" if muted else "Panic mute (Ctrl+Shift+M)"
        self.panic.setText(panic)
        self.setStyleSheet("QGroupBox { border: 2px solid #b00020; }" if muted else "")
        self.panic.setStyleSheet("background: #b00020; color: white;" if muted else "")
        speaker = self.session.settings.speaker_name
        error = self.session.path.monitor_error
        if self.monitor.isChecked() and error:
            self.monitor_warn.setText(error)
            self.monitor_warn.setVisible(True)
        else:
            self.monitor_warn.setText(
                "Monitor is speakers — the Character Voice may feed back "
                "into the mic."
            )
            self.monitor_warn.setVisible(
                self.monitor.isChecked() and monitor_looks_like_speakers(speaker)
            )
        self.hold_btn.setVisible(self.hold.isChecked())

    def toggle_panic(self) -> None:
        if not self.session.roleplay_on:
            return
        self.session.path.toggle_panic()
        self.refresh()

    def _toggle(self, checked: bool) -> None:
        if checked:
            try:
                self.session.start_roleplay()
                self.reason.setText("")
            except (RoleplayError, AudioDeviceError) as exc:
                self.session.stop_roleplay()
                self.reason.setText(str(exc))
                self.toggle.setChecked(False)
        else:
            self.session.stop_roleplay()
            self.reason.setText("")
        self.refresh()
        parent = self.window()
        if hasattr(parent, "refresh_chrome"):
            parent.refresh_chrome()

    def _hold_mode(self, checked: bool) -> None:
        self.session.path.set_hold_to_talk(checked)
        self.session.settings.save(self.session.store.root.parent)
        if not checked:
            self.session.path.set_talk_held(False)
        self.refresh()

    def _monitor(self, checked: bool) -> None:
        self.session.path.set_monitor(checked)
        self.session.settings.save(self.session.store.root.parent)
        self.refresh()

    def _tick(self) -> None:
        if self.session.roleplay_on:
            self.session.path.recover()
        peak_in = int(min(100, self.session.path.input_peak() * 100))
        peak_out = int(min(100, self.session.path.output_peak() * 100))
        self.in_meter.setValue(peak_in)
        self.out_meter.setValue(peak_out)
        receiving = (
            self.session.roleplay_on
            and not self.session.path.is_muted()
            and self.session.path.output_peak() >= _RECEIVING_PEAK
        )
        self.discord_hint.setText("Discord is receiving" if receiving else "")
        underruns, overruns, xruns = self.session.path.xrun_counts()
        self.xruns.setText(
            f"Underruns {underruns} · Overruns {overruns} · xruns {xruns}"
        )
        self.refresh()

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() == Qt.Key.Key_Alt and not event.isAutoRepeat():
            self.session.path.set_talk_held(True)
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event) -> None:  # noqa: N802
        if event.key() == Qt.Key.Key_Alt and not event.isAutoRepeat():
            self.session.path.set_talk_held(False)
        super().keyReleaseEvent(event)

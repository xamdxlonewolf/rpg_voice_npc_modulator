# Copyright (C) 2026 Michael Cobb
# SPDX-License-Identifier: GPL-3.0-or-later

"""Settings: devices, latency, hotkeys, Monitor, data folder, About."""

from __future__ import annotations

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices, QKeySequence
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QKeySequenceEdit,
    QLabel,
    QPushButton,
    QSlider,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from votr import WINDOW_TITLE, __version__
from votr.devices import find_cable_input, mic_devices
from votr.latency import format_latency_report
from votr.live import query_devices
from votr.neural import detect_nvidia_gpu, neural_status
from votr.paths import license_path, notices_path
from votr.preview import speaker_devices
from votr.session import Session
from votr.store import default_data_dir


class SettingsDialog(QDialog):
    def __init__(self, session: Session, parent=None) -> None:
        super().__init__(parent)
        self.session = session
        self.setWindowTitle("Settings")
        self.setObjectName("settings_dialog")
        self.resize(520, 480)
        root = QVBoxLayout(self)
        tabs = QTabWidget()
        tabs.addTab(self._devices_tab(), "Devices")
        tabs.addTab(self._latency_tab(), "Latency")
        tabs.addTab(self._general_tab(), "General")
        tabs.addTab(self._neural_tab(), "Neural")
        tabs.addTab(self._about_tab(), "About")
        root.addWidget(tabs)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        close = buttons.button(QDialogButtonBox.StandardButton.Close)
        if close is not None:
            close.clicked.connect(self._save)
        root.addWidget(buttons)

    def _devices_tab(self) -> QWidget:
        page = QWidget()
        layout = QFormLayout(page)
        self.mic = QComboBox()
        self.speakers = QComboBox()
        devices = query_devices()
        mics = mic_devices(devices)
        if not mics:
            self.mic.addItem("No microphone", "")
        for device in mics:
            self.mic.addItem(str(device["name"]), str(device["name"]))
        speakers = speaker_devices(devices)
        if not speakers:
            self.speakers.addItem("No speakers", "")
        for device in speakers:
            self.speakers.addItem(str(device["name"]), str(device["name"]))
        if self.session.settings.mic_name:
            index = self.mic.findData(self.session.settings.mic_name)
            if index >= 0:
                self.mic.setCurrentIndex(index)
        if self.session.settings.speaker_name:
            index = self.speakers.findData(self.session.settings.speaker_name)
            if index >= 0:
                self.speakers.setCurrentIndex(index)
        cable = find_cable_input(
            devices, preferred=self.session.settings.cable_input_name
        )
        cable_name = cable["name"] if cable else "none — open Discord setup"
        layout.addRow("Microphone", self.mic)
        layout.addRow("Speakers", self.speakers)
        layout.addRow("Virtual Cable", QLabel(str(cable_name)))
        return page

    def _latency_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        self.latency_report = QLabel(self._report_text())
        self.latency_report.setWordWrap(True)
        layout.addWidget(self.latency_report)
        rerun = QPushButton("Re-run test")
        rerun.clicked.connect(self._rerun)
        layout.addWidget(rerun)
        layout.addWidget(QLabel("Lower latency ↔ Cleaner sound"))
        self.latency_slider = QSlider(Qt.Orientation.Horizontal)
        self.latency_slider.setRange(0, 2)
        self.latency_slider.setValue(self.session.settings.latency_quality)
        self.latency_slider.valueChanged.connect(self._override)
        layout.addWidget(self.latency_slider)
        layout.addStretch()
        return page

    def _general_tab(self) -> QWidget:
        page = QWidget()
        layout = QFormLayout(page)
        self.hotkey = QKeySequenceEdit(QKeySequence(self.session.settings.panic_hotkey))
        layout.addRow("Panic mute", self.hotkey)
        hint = QLabel(
            "In-app shortcut (available while Voice of the Realm is focused)."
        )
        hint.setWordWrap(True)
        layout.addRow("", hint)
        self.monitor = QCheckBox("Monitor Character Voice by default")
        self.monitor.setChecked(self.session.settings.monitor_on)
        layout.addRow(self.monitor)
        data = self.session.store.root.parent
        layout.addRow("Data folder", QLabel(str(data or default_data_dir())))
        open_voices = QPushButton("Open Voices folder")
        open_voices.clicked.connect(self._open_voices)
        layout.addRow(open_voices)
        return page

    def _neural_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        self.neural_report = QLabel(neural_status(detect_nvidia_gpu()))
        self.neural_report.setWordWrap(True)
        self.neural_report.setObjectName("neural_report")
        self.neural_report.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        layout.addWidget(self.neural_report)
        layout.addStretch()
        return page

    def _about_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(QLabel(f"{WINDOW_TITLE} {__version__}"))
        layout.addWidget(QLabel("Licensed under GNU GPL-3.0-or-later."))
        notices = notices_path()
        license_file = license_path()
        if notices is not None:
            link = QPushButton("Third-party notices")
            link.clicked.connect(
                lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(notices)))
            )
            layout.addWidget(link)
        if license_file is not None:
            lic = QPushButton("LICENSE")
            lic.clicked.connect(
                lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(license_file)))
            )
            layout.addWidget(lic)
        layout.addStretch()
        return page

    def _report_text(self) -> str:
        settings = self.session.settings
        if not settings.latency_method:
            return "Latency Test has not been run. Re-run to measure this machine."
        from votr.latency import LatencyProbe

        probe = LatencyProbe(
            block_size=settings.block_size or 256,
            method=settings.latency_method,
            measured_ms=settings.latency_ms,
            engine_ms=settings.latency_engine_ms,
            device_ms=settings.latency_device_ms,
            total_ms=settings.latency_ms or settings.latency_engine_ms,
            underruns=None,
            blocked=settings.latency_blocked or None,
        )
        return format_latency_report(probe)

    def _rerun(self) -> None:
        duration = 10.0 if query_devices() else 0.0
        probe = self.session.run_latency_test(duration_s=duration)
        self.latency_report.setText(format_latency_report(probe))
        self.latency_slider.blockSignals(True)
        self.latency_slider.setValue(self.session.settings.latency_quality)
        self.latency_slider.blockSignals(False)

    def _override(self, quality: int) -> None:
        self.session.apply_latency_quality(quality)

    def _open_voices(self) -> None:
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.session.store.root)))

    def _save(self) -> None:
        self.session.settings.mic_name = str(self.mic.currentData() or "")
        self.session.settings.speaker_name = str(self.speakers.currentData() or "")
        cable = find_cable_input()
        if cable is not None:
            self.session.settings.cable_input_name = str(cable["name"])
        self.session.settings.monitor_on = self.monitor.isChecked()
        sequence = self.hotkey.keySequence().toString()
        if sequence:
            self.session.settings.panic_hotkey = sequence
        self.session.save_settings()

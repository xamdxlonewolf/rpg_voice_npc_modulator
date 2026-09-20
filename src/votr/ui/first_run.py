# Copyright (C) 2026 Michael Cobb
# SPDX-License-Identifier: GPL-3.0-or-later

"""First-run: welcome → devices → Latency Test → Discord → Library."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWizard,
    QWizardPage,
)

from votr.latency import format_latency_report
from votr.session import Session
from votr.ui.wizard import CablePage, DevicesPage, DiscordPage, SummaryPage


class WelcomePage(QWizardPage):
    def __init__(self) -> None:
        super().__init__()
        self.setTitle("Welcome")
        self.setSubTitle("Voice of the Realm")
        body = QLabel(
            "This setup picks a microphone and speakers, looks for a Virtual "
            "Cable (it never installs one), runs a Latency Test, and shows "
            "how Discord should use CABLE Output. You can skip and do this "
            "later from Settings."
        )
        body.setWordWrap(True)
        QVBoxLayout(self).addWidget(body)


class LatencyPage(QWizardPage):
    def __init__(self, session: Session) -> None:
        super().__init__()
        self.session = session
        self.setTitle("Latency Test")
        self.setSubTitle("Default target is about 120 ms. No invented numbers.")
        layout = QVBoxLayout(self)
        self.report = QLabel("The test has not been run yet.")
        self.report.setWordWrap(True)
        layout.addWidget(self.report)
        run = QPushButton("Run Latency Test")
        run.clicked.connect(self.run_test)
        layout.addWidget(run)
        layout.addWidget(QLabel("Lower latency ↔ Cleaner sound"))
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 2)
        self.slider.setValue(session.settings.latency_quality)
        self.slider.valueChanged.connect(self._override)
        layout.addWidget(self.slider)

    def initializePage(self) -> None:  # noqa: N802 — Qt
        self._show_saved()

    def run_test(self) -> None:
        from votr.live import query_devices

        duration = 10.0 if query_devices() else 0.0
        probe = self.session.run_latency_test(duration_s=duration)
        self.report.setText(format_latency_report(probe))
        self.slider.blockSignals(True)
        self.slider.setValue(self.session.settings.latency_quality)
        self.slider.blockSignals(False)

    def _show_saved(self) -> None:
        settings = self.session.settings
        if not settings.latency_method:
            return
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
        self.report.setText(format_latency_report(probe))

    def _override(self, quality: int) -> None:
        self.session.apply_latency_quality(quality)


class FirstRunWizard(QWizard):
    def __init__(self, session: Session, parent=None) -> None:
        super().__init__(parent)
        self.session = session
        self.setWindowTitle("First-run setup")
        self.setObjectName("first_run_wizard")
        self.setButtonText(QWizard.WizardButton.CancelButton, "Skip")
        self.welcome_page = WelcomePage()
        self.cable_page = CablePage(session)
        self.devices_page = DevicesPage(session)
        self.latency_page = LatencyPage(session)
        self.discord_page = DiscordPage(session)
        self.summary_page = SummaryPage(session)
        self.addPage(self.welcome_page)
        self.addPage(self.cable_page)
        self.addPage(self.devices_page)
        self.addPage(self.latency_page)
        self.addPage(self.discord_page)
        self.addPage(self.summary_page)
        self.accepted.connect(self._finished)
        self.rejected.connect(self._finished)

    def _finished(self) -> None:
        self.devices_page.apply()
        self.session.settings.first_run_completed = True
        self.session.settings.wizard_completed = True
        self.session.save_settings()

# Copyright (C) 2026 Michael Cobb
# SPDX-License-Identifier: GPL-3.0-or-later

"""Discord setup wizard: detect/link a Virtual Cable, pick mic and speakers."""

from __future__ import annotations

import sys

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QComboBox,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWizard,
    QWizardPage,
)

from votr.devices import (
    BLACKHOLE_URL,
    VB_CABLE_URL,
    create_linux_null_sink,
    device_ref,
    find_cable_input,
    find_cable_output,
    find_mic,
    find_speakers,
    has_virtual_cable,
    make_test_tone,
    mic_devices,
    play_to_cable,
)
from votr.live import query_devices
from votr.preview import speaker_devices
from votr.session import Session


def _open_url(url: str) -> None:
    QDesktopServices.openUrl(QUrl(url))


class CablePage(QWizardPage):
    def __init__(self, session: Session) -> None:
        super().__init__()
        self.session = session
        self.setTitle("Virtual Cable")
        self.setSubTitle(
            "Roleplay Mode sends the Character Voice into a Virtual Cable."
        )
        layout = QVBoxLayout(self)
        self.status = QLabel()
        self.status.setWordWrap(True)
        self.status.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        self.status.setOpenExternalLinks(True)
        layout.addWidget(self.status)
        again = QPushButton("Check again")
        again.clicked.connect(self.refresh)
        layout.addWidget(again)
        self.create_btn = QPushButton("Create VoiceOfTheRealm null sink (pactl)")
        self.create_btn.clicked.connect(self._create_sink)
        self.create_btn.setVisible(sys.platform.startswith("linux"))
        layout.addWidget(self.create_btn)
        vb = QPushButton("Download VB-CABLE (not bundled)")
        vb.clicked.connect(lambda: _open_url(VB_CABLE_URL))
        layout.addWidget(vb)
        hole = QPushButton("Download BlackHole (macOS, not bundled)")
        hole.clicked.connect(lambda: _open_url(BLACKHOLE_URL))
        hole.setVisible(sys.platform == "darwin")
        layout.addWidget(hole)
        self.refresh()

    def refresh(self) -> None:
        devices = query_devices()
        cable_in = find_cable_input(devices)
        cable_out = find_cable_output(devices)
        if cable_in is not None:
            out_name = cable_out["name"] if cable_out is not None else "not seen yet"
            self.status.setText(
                f"Virtual Cable found: {cable_in['name']}. "
                f"Discord should use: {out_name}."
            )
            self.session.settings.cable_input_name = str(cable_in["name"])
            return
        self.status.setText(
            "No Virtual Cable detected. The app never installs a driver. "
            f'<a href="{VB_CABLE_URL}">Download VB-CABLE</a>, install it, '
            "reboot Windows, then Check again. "
            "On Linux, create a null sink with pactl. "
            f'On macOS, install <a href="{BLACKHOLE_URL}">BlackHole</a>.'
        )

    def _create_sink(self) -> None:
        ok, detail = create_linux_null_sink()
        self.refresh()
        if not ok:
            self.status.setText(f"Could not create a null sink: {detail}")
        elif not has_virtual_cable():
            self.status.setText(
                f"Null sink created ({detail}). PortAudio still sees no cable."
            )


class DevicesPage(QWizardPage):
    def __init__(self, session: Session) -> None:
        super().__init__()
        self.session = session
        self.setTitle("Microphone and speakers")
        self.setSubTitle("The Virtual Cable is chosen automatically.")
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Microphone"))
        self.mic = QComboBox()
        layout.addWidget(self.mic)
        layout.addWidget(QLabel("Speakers (Preview and Monitor)"))
        self.speakers = QComboBox()
        layout.addWidget(self.speakers)
        self.cable = QLabel()
        self.cable.setWordWrap(True)
        layout.addWidget(self.cable)

    def initializePage(self) -> None:  # noqa: N802 — Qt
        devices = query_devices()
        self.mic.clear()
        mics = mic_devices(devices)
        if not mics:
            self.mic.addItem("No microphone", "")
        for device in mics:
            self.mic.addItem(str(device["name"]), str(device["name"]))
        if self.session.settings.mic_name:
            index = self.mic.findData(self.session.settings.mic_name)
            if index >= 0:
                self.mic.setCurrentIndex(index)
        self.speakers.clear()
        speakers = speaker_devices(devices)
        if not speakers:
            self.speakers.addItem("No speakers", "")
        for device in speakers:
            self.speakers.addItem(str(device["name"]), str(device["name"]))
        if self.session.settings.speaker_name:
            index = self.speakers.findData(self.session.settings.speaker_name)
            if index >= 0:
                self.speakers.setCurrentIndex(index)
        cable = find_cable_input(
            devices, preferred=self.session.settings.cable_input_name
        )
        if cable is None:
            self.cable.setText("Virtual Cable: none — Roleplay Mode cannot start.")
        else:
            self.cable.setText(f"Virtual Cable (automatic): {cable['name']}")

    def validatePage(self) -> bool:  # noqa: N802 — Qt
        self.apply()
        return True

    def apply(self) -> None:
        self.session.settings.mic_name = str(self.mic.currentData() or "")
        self.session.settings.speaker_name = str(self.speakers.currentData() or "")
        cable = find_cable_input(query_devices())
        if cable is not None:
            self.session.settings.cable_input_name = str(cable["name"])


class DiscordPage(QWizardPage):
    def __init__(self, session: Session) -> None:
        super().__init__()
        self.session = session
        self.setTitle("Discord input")
        layout = QVBoxLayout(self)
        self.hint = QLabel(
            "In Discord → Settings → Voice & Video → Input Device → "
            "CABLE Output (Windows) or VoiceOfTheRealm.monitor (Linux)."
        )
        self.hint.setWordWrap(True)
        layout.addWidget(self.hint)
        send = QPushButton("Send test tone/Take to cable")
        send.clicked.connect(self._send)
        layout.addWidget(send)
        self.result = QLabel()
        self.result.setWordWrap(True)
        layout.addWidget(self.result)

    def _send(self) -> None:
        devices = query_devices()
        cable = find_cable_input(
            devices, preferred=self.session.settings.cable_input_name
        )
        if cable is None:
            self.result.setText("No Virtual Cable — nothing was sent.")
            return
        take = self.session.take
        samples = take if take is not None else make_test_tone()
        played = play_to_cable(samples, device=device_ref(cable))
        if played:
            self.result.setText(
                "Playing to the Virtual Cable. Watch Discord's input meter."
            )
        else:
            self.result.setText(
                "Could not play to the cable (no device or PortAudio refused)."
            )


class SummaryPage(QWizardPage):
    def __init__(self, session: Session) -> None:
        super().__init__()
        self.session = session
        self.setTitle("Summary")
        self.body = QLabel()
        self.body.setWordWrap(True)
        QVBoxLayout(self).addWidget(self.body)

    def initializePage(self) -> None:  # noqa: N802 — Qt
        settings = self.session.settings
        cable = find_cable_input(preferred=settings.cable_input_name)
        mic = find_mic(preferred=settings.mic_name)
        speakers = find_speakers(preferred=settings.speaker_name)
        self.body.setText(
            "Mic: "
            + (mic["name"] if mic else settings.mic_name or "none")
            + "\nSpeakers: "
            + (speakers["name"] if speakers else settings.speaker_name or "none")
            + "\nVirtual Cable: "
            + (cable["name"] if cable else "none")
            + "\nRe-run this wizard from Settings → Discord setup."
        )


class DiscordWizard(QWizard):
    def __init__(self, session: Session, parent=None) -> None:
        super().__init__(parent)
        self.session = session
        self.setWindowTitle("Discord setup")
        self.setObjectName("discord_wizard")
        self.cable_page = CablePage(session)
        self.devices_page = DevicesPage(session)
        self.discord_page = DiscordPage(session)
        self.summary_page = SummaryPage(session)
        self.addPage(self.cable_page)
        self.addPage(self.devices_page)
        self.addPage(self.discord_page)
        self.addPage(self.summary_page)
        self.accepted.connect(self._save)

    def _save(self) -> None:
        self.devices_page.apply()
        self.session.settings.wizard_completed = True
        self.session.settings.save(self.session.store.root.parent)

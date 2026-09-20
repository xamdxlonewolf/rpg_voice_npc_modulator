# Copyright (C) 2026 Michael Cobb
# SPDX-License-Identifier: GPL-3.0-or-later

"""Roleplay Mode chrome. Live duplex lands in E4."""

from __future__ import annotations

from PySide6.QtWidgets import QGroupBox, QLabel, QVBoxLayout

from votr.session import Session


class RoleplayPanel(QGroupBox):
    def __init__(self, session: Session) -> None:
        super().__init__("Roleplay Mode")
        self.session = session
        self.setObjectName("roleplay_panel")
        layout = QVBoxLayout(self)
        self.active_label = QLabel()
        self.state_label = QLabel("Off — only the Character Voice will reach Discord.")
        layout.addWidget(self.active_label)
        layout.addWidget(self.state_label)
        self.refresh()

    def refresh(self) -> None:
        active = self.session.active_voice()
        name = active.name if active is not None else "none"
        self.active_label.setText(f"Active Voice: {name}")
        self.state_label.setText(
            "On" if self.session.roleplay_on else "Off — Character Voice only."
        )

# Copyright (C) 2026 Michael Cobb
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

from pathlib import Path

import pytest

from votr.devices import VB_CABLE_URL
from votr.session import Session
from votr.spikes.pitch_core import stretch_available


def _require_qt() -> None:
    try:
        from PySide6.QtWidgets import QApplication  # noqa: F401
    except (ImportError, OSError) as exc:
        pytest.skip(f"PySide6 unavailable: {exc}")


@pytest.mark.skipif(not stretch_available(), reason="DSP Engine needs python-stretch")
def test_roleplay_panel_points_at_wizard_without_cable(tmp_path: Path) -> None:
    _require_qt()
    from votr.app import create_application, create_main_window
    from votr.live import query_devices

    create_application(["votr-roleplay"])
    window = create_main_window(Session(tmp_path))
    window.show()
    assert "Active Voice" in window.roleplay.active_label.text()
    assert window.roleplay.wizard_btn.text().startswith("Discord")
    if query_devices():
        pytest.skip("audio devices present")
    window.roleplay.toggle.click()
    assert window.session.roleplay_on is False
    assert "Virtual Cable" in window.roleplay.reason.text()
    assert window.menuBar().actions()[0].text() == "Settings"


@pytest.mark.skipif(not stretch_available(), reason="DSP Engine needs python-stretch")
def test_wizard_links_vb_cable_never_bundled(tmp_path: Path) -> None:
    _require_qt()
    from votr.app import create_application
    from votr.ui.wizard import DiscordWizard

    create_application(["votr-wizard"])
    wizard = DiscordWizard(Session(tmp_path))
    assert len(wizard.pageIds()) == 4
    assert wizard.windowTitle().startswith("Voice of the Realm")
    assert "1a1a1a" in wizard.styleSheet()
    html = wizard.cable_page.status.text()
    assert VB_CABLE_URL in html
    assert "VB-CABLE" in html
    wizard.devices_page.initializePage()
    wizard.summary_page.initializePage()
    assert "CABLE Output" in wizard.discord_page.hint.text()


@pytest.mark.skipif(not stretch_available(), reason="DSP Engine needs python-stretch")
def test_monitor_default_off_and_panic_idle(tmp_path: Path) -> None:
    _require_qt()
    from votr.app import create_application, create_main_window

    create_application(["votr-monitor"])
    window = create_main_window(Session(tmp_path))
    assert window.roleplay.monitor.isChecked() is False
    window.roleplay.monitor.setChecked(True)
    assert "feed back" in window.roleplay.monitor_warn.text()
    window.roleplay.toggle_panic()
    assert window.session.path.is_muted() is False

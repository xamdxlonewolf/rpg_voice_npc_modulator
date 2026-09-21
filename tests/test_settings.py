# Copyright (C) 2026 Michael Cobb
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

from pathlib import Path

import pytest

from votr.session import Session
from votr.spikes.pitch_core import stretch_available


def _require_qt() -> None:
    try:
        from PySide6.QtWidgets import QApplication  # noqa: F401
    except (ImportError, OSError) as exc:
        pytest.skip(f"PySide6 unavailable: {exc}")


@pytest.mark.skipif(not stretch_available(), reason="DSP Engine needs python-stretch")
def test_first_run_wizard_pages_and_skip(tmp_path: Path) -> None:
    _require_qt()
    from votr.app import create_application
    from votr.ui.first_run import FirstRunWizard

    create_application(["votr-first-run"])
    session = Session(tmp_path)
    wizard = FirstRunWizard(session)
    assert len(wizard.pageIds()) == 6
    assert wizard.welcome_page.title() == "Voice of the Realm"
    assert "Character Voice" in wizard.welcome_page.body.text()
    assert "VB-CABLE" in wizard.welcome_page.body.text()
    assert wizard.windowTitle().startswith("Voice of the Realm")
    assert "1a1a1a" in wizard.styleSheet()
    wizard.latency_page.run_test()
    assert "Not glass-to-glass" in wizard.latency_page.report.text()
    wizard._finished()
    assert session.settings.first_run_completed is True
    assert Session(tmp_path).settings.first_run_completed is True


@pytest.mark.skipif(not stretch_available(), reason="DSP Engine needs python-stretch")
def test_settings_dialog_has_about_and_latency(tmp_path: Path) -> None:
    _require_qt()
    from votr.app import create_application, create_main_window
    from votr.ui.settings import SettingsDialog

    create_application(["votr-settings"])
    session = Session(tmp_path)
    window = create_main_window(session)
    dialog = SettingsDialog(session, window)
    assert dialog.objectName() == "settings_dialog"
    dialog._rerun()
    assert "glass-to-glass" in dialog.latency_report.text().lower()
    assert "not" in dialog.latency_report.text().lower()
    names = [action.text() for action in window.menuBar().actions()[0].menu().actions()]
    assert "Settings…" in names
    assert "First-run setup…" in names
    dialog._save()
    assert session.settings.panic_hotkey
    assert dialog.mic_gain.slider.objectName() == "mic_gain"
    assert session.settings.mic_gain_db == 0.0
    from votr.gain import db_to_slider, slider_to_db

    dialog.mic_gain.slider.setValue(db_to_slider(6.0))
    assert session.settings.mic_gain_db == pytest.approx(
        slider_to_db(db_to_slider(6.0))
    )
    assert Session(tmp_path).settings.mic_gain_db == pytest.approx(
        session.settings.mic_gain_db
    )
    about = dialog.findChild(type(dialog.neural_report), "about_install_note")
    assert about is not None
    assert "DSP" in about.text()
    assert "[neural]" in about.text()


@pytest.mark.skipif(not stretch_available(), reason="DSP Engine needs python-stretch")
def test_headless_run_skips_first_run_modal() -> None:
    _require_qt()
    from votr.app import run

    assert run(["--headless"]) == 0

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
def test_editor_saves_voice_to_store(tmp_path: Path) -> None:
    _require_qt()
    from votr.app import create_application, create_main_window

    create_application(["votr-editor-test"])
    session = Session(tmp_path)
    window = create_main_window(session)
    window.editor.name_edit.setText("Grimjaw")
    window.editor.hints_edit.setPlainText("orc captain")
    session.apply_tag("gravelly")
    window.editor.reload_from_draft()
    window.editor.save()
    voices = session.store.load_all()
    assert len(voices) == 1
    assert voices[0].name == "Grimjaw"
    assert voices[0].tone_hints == "orc captain"
    assert "gravelly" in voices[0].tone_tags
    assert session.active_voice() is not None
    assert window.header.text() == "Active Voice: Grimjaw"


def test_window_still_titled_voice_of_the_realm(tmp_path: Path) -> None:
    _require_qt()
    from votr import WINDOW_TITLE
    from votr.app import create_application, create_main_window

    create_application(["votr-title"])
    window = create_main_window(Session(tmp_path))
    assert window.windowTitle() == WINDOW_TITLE


@pytest.mark.skipif(not stretch_available(), reason="DSP Engine needs python-stretch")
def test_editor_preset_and_design_buttons(tmp_path: Path) -> None:
    _require_qt()
    from votr.app import create_application, create_main_window

    create_application(["votr-preset-test"])
    session = Session(tmp_path)
    window = create_main_window(session)
    editor = window.editor
    index = editor.preset_box.findData("Pixie")
    assert index > 0
    editor.preset_box.setCurrentIndex(index)
    editor._use_preset()
    assert editor.name_edit.text() == "Pixie"
    assert session.engine.params()["pitch_semitones"] == pytest.approx(7.0)
    assert "Pixie" in editor.design_status.text()

    editor.hints_edit.setPlainText("a whispering Irish ghost")
    editor.design_from_hints()
    assert "ghostly" in session.draft.tone_tags
    assert "whisper" in session.draft.tone_tags
    status = editor.design_status.text()
    assert "Tags:" in status
    assert "accent" in status.lower()
    assert session.engine.params()["breath"] > 0.0


def test_settings_has_an_honest_neural_tab(tmp_path: Path) -> None:
    _require_qt()
    from votr.app import create_application
    from votr.ui.settings import SettingsDialog

    create_application(["votr-neural-tab"])
    dialog = SettingsDialog(Session(tmp_path))
    text = dialog.neural_report.text()
    assert "Neural Engine" in text
    assert "not installed" in text

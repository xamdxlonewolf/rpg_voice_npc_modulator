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

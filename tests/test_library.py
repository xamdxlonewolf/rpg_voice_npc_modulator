# Copyright (C) 2026 Michael Cobb
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

from pathlib import Path

import pytest

from votr.session import Session
from votr.spikes.pitch_core import stretch_available
from votr.voice import Voice


def _require_qt() -> None:
    try:
        from PySide6.QtWidgets import QApplication  # noqa: F401
    except (ImportError, OSError) as exc:
        pytest.skip(f"PySide6 unavailable: {exc}")


@pytest.mark.skipif(not stretch_available(), reason="DSP Engine needs python-stretch")
def test_library_empty_state_and_new_voice(tmp_path: Path) -> None:
    _require_qt()
    from votr.app import create_application, create_main_window

    create_application(["votr-lib"])
    window = create_main_window(Session(tmp_path))
    window.show()
    assert not window.library.empty.isHidden()
    assert "first Voice" in window.library.empty.text()
    window.show_new_editor()
    window.editor.name_edit.setText("Grimjaw")
    window.editor.save()
    assert window.pages.currentWidget() is window.library
    assert not window.library.empty.isVisible()
    assert window.header.text() == "Active Voice: Grimjaw"
    assert "Grimjaw" in window.roleplay.active_label.text()


@pytest.mark.skipif(not stretch_available(), reason="DSP Engine needs python-stretch")
def test_search_and_disabled_engine(tmp_path: Path) -> None:
    _require_qt()
    from votr.app import create_application, create_main_window
    from votr.ui.library import VoiceButton

    session = Session(tmp_path)
    grim = Voice.new("Grimjaw")
    grim.tone_tags = ["gravelly"]
    session.store.save(grim)
    other = Voice.new("Neural Elf")
    other.engine_id = "neural-v1"
    session.store.save(other)
    session.voices = session.store.load_all()
    create_application(["votr-lib-2"])
    window = create_main_window(session)
    window.show()
    window.library.search.setText("gravel")
    window.library.refresh()
    from PySide6.QtWidgets import QApplication

    QApplication.processEvents()
    buttons = [
        button
        for button in window.library.findChildren(VoiceButton)
        if button.parent() is window.library._grid_host
    ]
    assert [button.text().split("\n")[0] for button in buttons] == ["Grimjaw"]
    window.library.search.clear()
    window.library.refresh()
    from PySide6.QtWidgets import QApplication

    QApplication.processEvents()
    neural = next(
        button
        for button in window.library.findChildren(VoiceButton)
        if button.parent() is window.library._grid_host and "Neural" in button.text()
    )
    assert not neural.isEnabled()
    assert "not installed" in neural.toolTip()

# Copyright (C) 2026 Michael Cobb
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import sys
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

    # Tune, Save, then pick the preset again: the saved Voice comes back.
    editor._sliders["pitch_semitones"].setValue(500)
    editor.save()
    window.show_editor(session.voices[0].id)
    assert session.draft.params["pitch_semitones"] == pytest.approx(0.0)
    editor.preset_box.setCurrentIndex(editor.preset_box.findData("Pixie"))
    assert "saved" in editor.preset_box.currentText()
    editor._use_preset()
    assert editor.name_edit.text() == "Pixie"
    assert session.engine.params()["pitch_semitones"] == pytest.approx(0.0)
    assert "saved" in editor.design_status.text()
    editor._fresh_preset()
    assert editor.name_edit.text() == "Pixie 2"
    assert session.engine.params()["pitch_semitones"] == pytest.approx(7.0)
    session.edit_new()
    editor.reload_from_draft()

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
    assert "not downloaded" in text
    assert "accent" in text.lower()
    # No GPU on this machine: packs are listed with size and licences, but the
    # download buttons stay off even after acknowledging the licences.
    button = dialog.pack_buttons["xvc"]
    assert "GB" in button.text()
    assert not button.isEnabled()
    dialog.licence_ack.setChecked(True)
    assert not button.isEnabled()
    assert "NVIDIA" in button.toolTip()
    assert dialog.pack_labels["xvc"].text() == "Not downloaded."
    assert not dialog.pack_progress.isVisible()


def test_settings_download_is_gated_and_runs_off_thread(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _require_qt()
    import hashlib

    from PySide6.QtCore import QCoreApplication, QEventLoop, QTimer

    from votr import neural as neural_module
    from votr.app import create_application
    from votr.neural import GpuInfo, probe_runtime
    from votr.neural_pack import ModelPack, PackFile
    from votr.ui import settings as settings_ui
    from votr.ui.settings import SettingsDialog

    sys.path.insert(0, str(Path(__file__).parent))
    from test_neural_pack import _RangeServer

    payload = b"weights" * 1000
    server = _RangeServer({"w.bin": payload})
    server.start()
    try:
        pack = ModelPack(
            "xvc",
            "Fake X-VC",
            "test",
            (
                PackFile(
                    "xvc/xvc.pt",
                    server.url + "w.bin",
                    len(payload),
                    hashlib.sha256(payload).hexdigest(),
                    "MIT",
                ),
            ),
            ("MIT",),
        )
        monkeypatch.setattr(settings_ui, "PACKS", (pack,))
        monkeypatch.setattr(neural_module, "CONVERSION_PACK", pack)
        monkeypatch.setattr(
            neural_module, "torch_cuda_state", lambda: (True, True, "ok")
        )
        create_application(["votr-neural-download"])
        session = Session(tmp_path)
        session.neural = probe_runtime(
            tmp_path,
            gpu=GpuInfo("RTX 3060", 12288),
            torch_state=lambda: (True, True, "ok"),
        )
        dialog = SettingsDialog(session)
        button = dialog.pack_buttons["xvc"]
        assert not button.isEnabled() and "licence" in button.toolTip().lower()
        dialog.licence_ack.setChecked(True)
        assert button.isEnabled()
        dialog._download("xvc")
        assert dialog.pack_cancel.isVisible() or dialog._worker is not None
        loop = QEventLoop()
        dialog._worker.finished.connect(loop.quit)
        QTimer.singleShot(10_000, loop.quit)
        if dialog._worker.isRunning():
            loop.exec()
        QCoreApplication.processEvents()
        assert (tmp_path / "neural" / "xvc" / "xvc.pt").read_bytes() == payload
        assert dialog.pack_labels["xvc"].text() == "Installed."
        assert not button.isEnabled() and button.text() == "Installed"
        assert "Download finished" in dialog.neural_report.text()
    finally:
        server.stop()

# Copyright (C) 2026 Michael Cobb
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from votr.neural import GpuInfo, probe_runtime
from votr.neural_engine import NeuralEngine
from votr.neural_pack import CONVERSION_PACK, pack_root
from votr.session import Session
from votr.spikes.pitch_core import stretch_available
from votr.voice import Voice

BIG_GPU = GpuInfo("NVIDIA GeForce RTX 3060", 12288)


def _require_qt() -> None:
    try:
        from PySide6.QtWidgets import QApplication  # noqa: F401
    except (ImportError, OSError) as exc:
        pytest.skip(f"PySide6 unavailable: {exc}")


def _installed_pack(root: Path, pack) -> None:
    for item in pack.files:
        if item.unzip_to:
            (root / item.unzip_to).mkdir(parents=True, exist_ok=True)
            (root / item.unzip_to / "README.md").write_text("x")
            continue
        path = root / item.relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            handle.truncate(item.size or 0)


class FakeConverter:
    sample_rate = 16000

    def set_reference(self, clip, sample_rate: int) -> None:
        return None

    def convert_window(self, window):
        return -window


@pytest.mark.skipif(not stretch_available(), reason="DSP Engine needs python-stretch")
def test_dsp_only_session_does_not_preload(tmp_path: Path) -> None:
    session = Session(tmp_path)
    assert session.should_preload_neural() is False
    assert session.neural.torch_detail.startswith("not checked")


@pytest.mark.skipif(not stretch_available(), reason="DSP Engine needs python-stretch")
def test_should_preload_when_pack_and_gpu_even_before_torch(
    tmp_path: Path,
) -> None:
    session = Session(tmp_path)
    _installed_pack(pack_root(tmp_path), CONVERSION_PACK)
    session.neural = probe_runtime(tmp_path, gpu=BIG_GPU, check_torch=False)
    assert session.neural.ready is False
    assert session.should_preload_neural() is True


@pytest.mark.skipif(not stretch_available(), reason="DSP Engine needs python-stretch")
def test_should_preload_for_saved_neural_voice_with_pack(
    tmp_path: Path,
) -> None:
    session = Session(tmp_path)
    _installed_pack(pack_root(tmp_path), CONVERSION_PACK)
    session.neural = probe_runtime(tmp_path, detect=lambda: None, check_torch=False)
    voice = Voice.new()
    voice.engine_id = "neural-v0"
    session.voices.append(voice)
    assert session.neural.gpu_ok is False
    assert session.should_preload_neural() is True


@pytest.mark.skipif(not stretch_available(), reason="DSP Engine needs python-stretch")
def test_ensure_returns_none_while_preload_is_active(tmp_path: Path) -> None:
    session = Session(tmp_path)
    session.begin_neural_preload()
    assert session.neural_loading
    assert session.ensure_neural_engine() is None
    session.end_neural_preload()
    assert not session.neural_loading


@pytest.mark.skipif(not stretch_available(), reason="DSP Engine needs python-stretch")
def test_preload_overlay_loads_off_ui_thread(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _require_qt()
    from PySide6.QtCore import QEventLoop, QTimer
    from PySide6.QtWidgets import QApplication

    import votr.session as session_module
    from votr.app import create_application, create_main_window
    from votr.ui.neural_preload import LOADING_TEXT

    create_application(["votr-neural-preload"])
    session = Session(tmp_path)
    _installed_pack(pack_root(tmp_path), CONVERSION_PACK)
    session.neural = probe_runtime(tmp_path, gpu=BIG_GPU, check_torch=False)

    def fake_refresh() -> None:
        session.neural = probe_runtime(
            tmp_path, gpu=BIG_GPU, torch_state=lambda: (True, True, "ok")
        )
        session.neural_error = ""

    ui_thread = threading.get_ident()
    seen: list[int] = []

    def fake_build(runtime, block_size=512):
        seen.append(threading.get_ident())
        time.sleep(0.15)
        return NeuralEngine(FakeConverter(), block_size=block_size)

    monkeypatch.setattr(session, "refresh_neural", fake_refresh)
    monkeypatch.setattr(session_module, "build_neural_engine", fake_build)

    window = create_main_window(session)
    window.show()
    QApplication.processEvents()
    assert window._neural_overlay is None

    worker = window.start_neural_preload()
    assert worker is not None
    overlay = window._neural_overlay
    assert overlay is not None
    assert overlay.isVisible()
    assert overlay.label.text() == LOADING_TEXT
    assert overlay.spinner.minimum() == overlay.spinner.maximum() == 0

    loop = QEventLoop()
    worker.finished.connect(loop.quit)
    QTimer.singleShot(10_000, loop.quit)
    if not worker.isFinished():
        loop.exec()
    QApplication.processEvents()

    assert seen and seen[0] != ui_thread
    assert session.neural_engine is not None
    assert window._neural_overlay is None

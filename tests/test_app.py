# Copyright (C) 2026 Michael Cobb
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import os

import pytest

from votr import WINDOW_TITLE, __version__


def _require_qt_widgets() -> None:
    """Skip GUI tests when Qt native libraries are missing (headless CI)."""
    try:
        from PySide6.QtWidgets import QApplication, QMainWindow, QWidget  # noqa: F401
    except (ImportError, OSError) as exc:
        pytest.skip(f"PySide6 QtWidgets unavailable: {exc}")


def test_package_version() -> None:
    assert __version__ == "0.1.0"
    assert WINDOW_TITLE == "Voice of the Realm"


def test_headless_flag_without_qt(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VOTR_HEADLESS", raising=False)
    from votr.app import is_headless

    assert is_headless(["--headless"]) is True
    assert is_headless([]) is False
    monkeypatch.setenv("VOTR_HEADLESS", "1")
    assert is_headless([]) is True


def test_window_title_offscreen() -> None:
    _require_qt_widgets()
    from votr.app import create_application, create_main_window

    create_application(["votr-test"])
    window = create_main_window()
    assert window.windowTitle() == WINDOW_TITLE


def test_run_headless_exits_zero() -> None:
    _require_qt_widgets()
    from votr.app import run

    assert run(["--headless"]) == 0


def test_run_headless_defaults_offscreen_platform(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("QT_QPA_PLATFORM", raising=False)
    _require_qt_widgets()
    from votr.app import run

    assert run(["--headless"]) == 0
    assert os.environ.get("QT_QPA_PLATFORM") == "offscreen"

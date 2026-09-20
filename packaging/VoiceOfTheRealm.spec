# -*- mode: python ; coding: utf-8 -*-
# Copyright (C) 2026 Michael Cobb
# SPDX-License-Identifier: GPL-3.0-or-later
"""PyInstaller --onedir spec. Does not collect-all PySide6 Addons."""

from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

ROOT = Path(SPECPATH).resolve().parent

datas = [
    (str(ROOT / "src" / "votr" / "assets"), "votr/assets"),
    (str(ROOT / "THIRD_PARTY_NOTICES.md"), "."),
    (str(ROOT / "LICENSE"), "."),
]
binaries = []
hiddenimports = [
    "PySide6.QtCore",
    "PySide6.QtGui",
    "PySide6.QtWidgets",
    "numpy",
    "sounddevice",
    "pedalboard",
    "python_stretch",
    "platformdirs",
    "votr",
    "votr.app",
    "votr.assets",
]

for package in ("sounddevice", "pedalboard", "python_stretch", "PySide6"):
    try:
        datas += collect_data_files(package)
    except Exception:
        pass
    try:
        binaries += collect_dynamic_libs(package)
    except Exception:
        pass

try:
    import rubband  # noqa: F401

    hiddenimports.append("rubband")
    datas += collect_data_files("rubband")
    binaries += collect_dynamic_libs("rubband")
except Exception:
    pass

# Qt platform plugins only — not PySide6 Addons (S0.6 Linux collect-all was 722 MB).
try:
    datas += collect_data_files(
        "PySide6",
        includes=[
            "plugins/platforms/*",
            "plugins/styles/*",
            "plugins/imageformats/*",
        ],
    )
except Exception:
    pass

a = Analysis(
    [str(ROOT / "src" / "votr" / "__main__.py")],
    pathex=[str(ROOT / "src")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[str(ROOT / "packaging" / "rthooks" / "pyi_rth_votr.py")],
    excludes=["tkinter", "matplotlib", "PySide6.QtCharts"],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="VoiceOfTheRealm",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="VoiceOfTheRealm",
)

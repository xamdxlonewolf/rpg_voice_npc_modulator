# Copyright (C) 2026 Michael Cobb
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import subprocess

import pytest

from votr import neural
from votr.neural import (
    MIN_VRAM_MIB,
    GpuInfo,
    NeuralDesigner,
    detect_nvidia_gpu,
    neural_status,
    parse_nvidia_smi,
)
from votr.session import INSTALLED_ENGINE_IDS


def test_parse_nvidia_smi_csv() -> None:
    gpu = parse_nvidia_smi("NVIDIA GeForce RTX 3060, 12288 MiB\n")
    assert gpu == GpuInfo(name="NVIDIA GeForce RTX 3060", vram_mib=12288)
    assert gpu.enough_vram
    assert parse_nvidia_smi("") is None
    assert parse_nvidia_smi("garbage line") is None
    small = parse_nvidia_smi("NVIDIA GeForce GTX 1050, 2048 MiB")
    assert small is not None and not small.enough_vram


def test_detect_without_nvidia_smi_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(neural.shutil, "which", lambda _name: None)
    assert detect_nvidia_gpu() is None


def test_detect_parses_a_fake_driver(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(neural.shutil, "which", lambda _name: "/fake/nvidia-smi")

    def fake_run(*_args, **_kwargs):
        return subprocess.CompletedProcess(
            args=[], returncode=0, stdout="NVIDIA RTX A4000, 16376 MiB\n", stderr=""
        )

    monkeypatch.setattr(neural.subprocess, "run", fake_run)
    gpu = detect_nvidia_gpu()
    assert gpu == GpuInfo("NVIDIA RTX A4000", 16376)


def test_detect_tolerates_a_broken_driver(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(neural.shutil, "which", lambda _name: "/fake/nvidia-smi")

    def boom(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=1)

    monkeypatch.setattr(neural.subprocess, "run", boom)
    assert detect_nvidia_gpu() is None


def test_status_is_honest_about_hardware_and_accent() -> None:
    none = neural_status(None)
    assert "No NVIDIA GPU" in none
    assert "not installed" in none
    assert "accent" in none.lower()
    small = neural_status(GpuInfo("GTX 1050", 2048))
    assert str(MIN_VRAM_MIB) in small
    big = neural_status(GpuInfo("RTX 3060", 12288))
    assert "enough" in big
    for text in (none, small, big):
        assert "download button" in text


def test_neural_designer_reports_unavailable_and_engine_is_not_installed() -> None:
    ok, reason = NeuralDesigner().available()
    assert not ok and "not installed" in reason
    ok, reason = NeuralDesigner(GpuInfo("RTX 3060", 12288), installed=True).available()
    assert not ok and "no model" in reason
    assert neural.NEURAL_ENGINE_ID not in INSTALLED_ENGINE_IDS

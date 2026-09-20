# Copyright (C) 2026 Michael Cobb
# SPDX-License-Identifier: GPL-3.0-or-later

"""Neural Engine seam (E7): GPU detection and honest availability.

Nothing here downloads or runs a model. It answers two questions the UI needs:
does this machine have an NVIDIA GPU with enough memory, and what exactly is
missing before a neural Voice designer or Engine could run. The DSP Engine is
always the fallback. See docs/neural-voice.md for what is and is not feasible.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass

from votr.voicedesign import VoiceDesign

NEURAL_ENGINE_ID = "neural-v0"
MIN_VRAM_MIB = 6 * 1024
DOWNLOAD_ESTIMATE = "roughly 5–8 GB (CUDA PyTorch plus model weights)"


@dataclass(frozen=True)
class GpuInfo:
    name: str
    vram_mib: int

    @property
    def enough_vram(self) -> bool:
        return self.vram_mib >= MIN_VRAM_MIB


def parse_nvidia_smi(output: str) -> GpuInfo | None:
    """Parse ``nvidia-smi --query-gpu=name,memory.total --format=csv,noheader``."""
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2:
            continue
        digits = "".join(ch for ch in parts[1] if ch.isdigit())
        if not digits:
            continue
        return GpuInfo(name=parts[0], vram_mib=int(digits))
    return None


def detect_nvidia_gpu(timeout_s: float = 3.0) -> GpuInfo | None:
    """First NVIDIA GPU reported by nvidia-smi, or None (no driver, no GPU)."""
    binary = shutil.which("nvidia-smi")
    if binary is None:
        return None
    try:
        result = subprocess.run(
            [binary, "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return parse_nvidia_smi(result.stdout)


def neural_status(gpu: GpuInfo | None) -> str:
    """Plain-language state of the Neural Engine on this machine."""
    if gpu is None:
        hardware = (
            "No NVIDIA GPU detected. The Neural Engine cannot run on this machine; "
            "the DSP Engine (all current sliders and presets) is CPU-only and works."
        )
    elif not gpu.enough_vram:
        hardware = (
            f"{gpu.name} with {gpu.vram_mib} MiB found. Zero-shot voice conversion "
            f"needs about {MIN_VRAM_MIB} MiB; expect it to be too tight."
        )
    else:
        hardware = (
            f"{gpu.name} with {gpu.vram_mib} MiB found — enough for the planned "
            "Neural Engine once it is installed."
        )
    return (
        f"{hardware}\n\n"
        "Not in this build: the Neural Engine and Neural Voice Design are not "
        f"installed and there is no download button yet. Installing means an opt-in "
        f"download of {DOWNLOAD_ESTIMATE} into your data folder, NVIDIA only.\n\n"
        "What it would add: sounding like a different person from a short reference "
        "clip, and describing a voice in words to get that person. What it would "
        "not add: changing your accent live. Voice conversion moves timbre, not "
        "pronunciation — an Irish or British accent still has to come from you."
    )


class NeuralDesigner:
    """Prompt → reference clip → Voice. Placeholder: reports why it is off."""

    designer_id = "neural"

    def __init__(self, gpu: GpuInfo | None = None, *, installed: bool = False) -> None:
        self._gpu = gpu
        self._installed = installed

    def available(self) -> tuple[bool, str]:
        if not self._installed:
            return False, (
                "Neural Voice Design is not installed in this build (needs an NVIDIA "
                f"GPU with ≥ {MIN_VRAM_MIB} MiB and an opt-in download of "
                f"{DOWNLOAD_ESTIMATE})."
            )
        if self._gpu is None or not self._gpu.enough_vram:
            return False, "Neural Voice Design needs an NVIDIA GPU with enough memory."
        return False, "Neural Voice Design has no model wired in yet."

    def design(self, prompt: str) -> VoiceDesign:
        raise RuntimeError(self.available()[1])

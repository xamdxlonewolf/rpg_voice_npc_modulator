# Copyright (C) 2026 Michael Cobb
# SPDX-License-Identifier: GPL-3.0-or-later

"""Neural Engine gate (E7): GPU detection, runtime probe, honest availability.

Three things must all be true before anything neural runs: an NVIDIA GPU with
enough memory, a CUDA build of PyTorch importable in this Python, and the model
pack downloaded (Settings → Neural, opt-in). Otherwise everything here reports
*why not* and the DSP Engine plus the lexicon designer remain the path.
"""

from __future__ import annotations

import importlib.util
import logging
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from votr.clips import REFERENCE_CLIP_ID_KEY, ClipLibrary
from votr.neural_engine import (
    NEURAL_ENGINE_ID,
    REFERENCE_CLIP_KEY,
    NeuralEngine,
    NeuralUnavailable,
)
from votr.neural_pack import (
    CONVERSION_PACK,
    VOICE_DESIGN_PACK,
    PackStatus,
    pack_root,
    pack_status,
)
from votr.voicedesign import VoiceDesign

__all__ = [
    "NEURAL_ENGINE_ID",
    "GpuInfo",
    "NeuralDesigner",
    "NeuralRuntime",
    "build_neural_engine",
    "create_neural_engine",
    "detect_nvidia_gpu",
    "neural_status",
    "parse_nvidia_smi",
    "probe_runtime",
]

log = logging.getLogger("votr.neural")

MIN_VRAM_MIB = 6 * 1024
DOWNLOAD_ESTIMATE = "about 6.5 GB for conversion, 4.5 GB more for voice design"


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


def torch_cuda_state() -> tuple[bool, bool, str]:
    """(torch importable, CUDA usable, detail). Never imports torch if absent."""
    if importlib.util.find_spec("torch") is None:
        return False, False, "PyTorch is not installed in this Python"
    try:
        import torch  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - broken install
        return False, False, f"PyTorch failed to import: {exc}"
    try:
        cuda = bool(torch.cuda.is_available())
    except Exception as exc:  # pragma: no cover
        return True, False, f"torch.cuda check failed: {exc}"
    version = getattr(torch, "__version__", "?")
    if not cuda:
        return (
            True,
            False,
            f"PyTorch {version} is installed without a usable CUDA device",
        )
    return True, True, f"PyTorch {version} with CUDA"


@dataclass(frozen=True)
class NeuralRuntime:
    gpu: GpuInfo | None
    torch_ok: bool
    cuda_ok: bool
    torch_detail: str
    conversion: PackStatus
    design: PackStatus
    root: Path

    @property
    def gpu_ok(self) -> bool:
        return self.gpu is not None and self.gpu.enough_vram

    @property
    def ready(self) -> bool:
        return self.gpu_ok and self.cuda_ok and self.conversion.installed

    @property
    def design_ready(self) -> bool:
        return self.ready and self.design.installed

    @property
    def reason(self) -> str:
        if self.gpu is None:
            return "no NVIDIA GPU detected"
        if not self.gpu.enough_vram:
            return f"{self.gpu.name} has {self.gpu.vram_mib} MiB; needs {MIN_VRAM_MIB}"
        if not self.cuda_ok:
            return self.torch_detail
        if not self.conversion.installed:
            return "the X-VC model pack is not downloaded"
        return ""


def probe_runtime(
    data_dir: Path,
    *,
    gpu: GpuInfo | None = None,
    detect: Callable[[], GpuInfo | None] = detect_nvidia_gpu,
    torch_state: Callable[[], tuple[bool, bool, str]] = torch_cuda_state,
) -> NeuralRuntime:
    root = pack_root(Path(data_dir))
    found = gpu if gpu is not None else detect()
    torch_ok, cuda_ok, detail = (
        torch_state()
        if found is not None
        else (
            False,
            False,
            "not checked (no NVIDIA GPU)",
        )
    )
    return NeuralRuntime(
        gpu=found,
        torch_ok=torch_ok,
        cuda_ok=cuda_ok,
        torch_detail=detail,
        conversion=pack_status(CONVERSION_PACK, root),
        design=pack_status(VOICE_DESIGN_PACK, root),
        root=root,
    )


ConverterFactory = Callable[[Path], Any]


def _default_converter(root: Path) -> Any:
    from votr.neural_backends import XvcConverter

    return XvcConverter(root)


def build_neural_engine(
    runtime: NeuralRuntime,
    *,
    block_size: int = 512,
    converter_factory: ConverterFactory = _default_converter,
) -> NeuralEngine:
    """Build the Neural Engine; raises ``NeuralUnavailable`` with the real reason."""
    if not runtime.ready:
        raise NeuralUnavailable(runtime.reason)
    try:
        converter = converter_factory(runtime.root)
        return NeuralEngine(converter, block_size=block_size)
    except NeuralUnavailable:
        raise
    except Exception as exc:
        raise NeuralUnavailable(f"{exc}") from exc


def create_neural_engine(
    runtime: NeuralRuntime,
    *,
    block_size: int = 512,
    converter_factory: ConverterFactory = _default_converter,
) -> NeuralEngine | None:
    """Build the Neural Engine, or None (with a log line) when it cannot run."""
    try:
        return build_neural_engine(
            runtime, block_size=block_size, converter_factory=converter_factory
        )
    except NeuralUnavailable as exc:
        if runtime.ready:
            log.warning("Neural Engine failed to start: %s", exc)
        else:
            log.info("Neural Engine unavailable: %s", exc)
        return None


def neural_status(gpu: GpuInfo | None, runtime: NeuralRuntime | None = None) -> str:
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
            f"{gpu.name} with {gpu.vram_mib} MiB found — enough for the Neural "
            "Engine once its model pack is downloaded."
        )
    if runtime is None:
        install = (
            "Not installed: the model packs are an opt-in download "
            f"({DOWNLOAD_ESTIMATE}) into your data folder, NVIDIA only. There is no "
            "silent download; you see size and licences first and can cancel."
        )
    else:
        conv = "installed" if runtime.conversion.installed else "not downloaded"
        design = "installed" if runtime.design.installed else "not downloaded"
        install = (
            f"Model packs: X-VC conversion {conv}; Qwen3-TTS VoiceDesign {design}. "
            f"PyTorch: {runtime.torch_detail}. "
            + (
                "Neural Engine is ready."
                if runtime.ready
                else f"Neural Engine is off — {runtime.reason}."
            )
        )
    return (
        f"{hardware}\n\n{install}\n\n"
        "What it adds: sounding like a different person from a short reference "
        "clip, and (with the VoiceDesign pack) describing a voice in words to get "
        "that person. What it does not add: changing your accent live. Voice "
        "conversion moves timbre, not pronunciation — an Irish or British accent "
        "still has to come from you."
    )


class NeuralDesigner:
    """Prompt → spoken reference clip → Neural Voice (when the packs are here)."""

    designer_id = "neural"

    def __init__(
        self,
        runtime: NeuralRuntime | None = None,
        *,
        clip_maker_factory: Callable[[Path], Any] | None = None,
    ) -> None:
        self._runtime = runtime
        self._factory = clip_maker_factory
        self._maker: Any = None

    def available(self) -> tuple[bool, str]:
        if self._runtime is None:
            return False, (
                "Neural Voice Design is not installed (needs an NVIDIA GPU with ≥ "
                f"{MIN_VRAM_MIB} MiB and the opt-in model packs, {DOWNLOAD_ESTIMATE})."
            )
        if not self._runtime.ready:
            return False, f"Neural Voice Design is off — {self._runtime.reason}."
        if not self._runtime.design.installed:
            return (
                False,
                "Neural Voice Design is off — the VoiceDesign pack is not downloaded.",
            )
        return True, ""

    def _clip_maker(self) -> Any:
        if self._maker is None:
            assert self._runtime is not None
            factory = self._factory or _default_clip_maker
            self._maker = factory(self._runtime.root)
        return self._maker

    def design(self, prompt: str) -> VoiceDesign:
        ok, reason = self.available()
        if not ok:
            raise RuntimeError(reason)
        assert self._runtime is not None
        from votr.voicedesign import _guess_name

        audio, rate = self._clip_maker().make_clip(prompt.strip())
        name = _guess_name(prompt)
        library = ClipLibrary(self._runtime.root.parent)
        clip = library.add(audio, int(rate), name, origin="designed")
        return VoiceDesign(
            name=name,
            tone_tags=[],
            params={
                REFERENCE_CLIP_ID_KEY: clip.id,
                REFERENCE_CLIP_KEY: str(library.path_for(clip.id)),
                "mix": 1.0,
            },
            matched=[prompt.strip()],
            notes=[
                "Reference clip spoken by Qwen3-TTS VoiceDesign from your words; the "
                "Neural Engine converts your voice toward it. Accent is still yours."
            ],
            designer=self.designer_id,
            engine_id=NEURAL_ENGINE_ID,
        )


def _default_clip_maker(root: Path) -> Any:
    from votr.neural_backends import QwenVoiceDesign

    return QwenVoiceDesign(root)

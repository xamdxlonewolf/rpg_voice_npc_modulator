# Copyright (C) 2026 Michael Cobb
# SPDX-License-Identifier: GPL-3.0-or-later

"""Model adapters for the Neural Engine and neural Voice design.

These wrap the publishers' own inference code (X-VC ``bins/infer_utils`` and
the ``qwen_tts`` package) so the rest of the app only sees ``VoiceConverter``
and ``ReferenceClipMaker``. They import torch lazily and only run when the
model packs are installed and an NVIDIA GPU is present.

Written against the upstream sources as of X-VC commit 49df8c5 and qwen-tts
0.1.1; **not yet exercised on real hardware by this project.** Expect the
first run on a GPU box to surface small API mismatches — keep changes here.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from votr.neural_pack import CONVERSION_PACK, VOICE_DESIGN_PACK, pack_status

XVC_LATENT_HOP = 1280
DESIGN_SAMPLE_TEXT = (
    "Gather close, travellers, and listen well: the road ahead winds through the "
    "old forest, and not everything that watches from the trees is a friend."
)


class ReferenceClipMaker(Protocol):
    def make_clip(
        self, instruct: str, text: str = DESIGN_SAMPLE_TEXT
    ) -> tuple[np.ndarray, int]:
        """Speak ``text`` in the voice described by ``instruct``; (mono, rate)."""


def _cuda_device(index: int = 0) -> Any:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available to PyTorch on this machine")
    return torch.device(f"cuda:{index}")


class XvcConverter:
    """``VoiceConverter`` over X-VC's streaming forward pass."""

    sample_rate = 16000

    def __init__(self, root: Path, *, device_index: int = 0) -> None:
        self._root = Path(root)
        status = pack_status(CONVERSION_PACK, self._root)
        if not status.installed:
            missing = ", ".join(item.relpath for item in status.missing)
            raise RuntimeError(f"X-VC pack incomplete: missing {missing}")
        source = self._root / "xvc-src"
        if str(source) not in sys.path:
            sys.path.insert(0, str(source))
        import torch
        from bins import infer_utils  # type: ignore[import-not-found]

        self._torch = torch
        self._infer = infer_utils
        self._device = _cuda_device(device_index)
        config_path = self._write_runtime_config(source)
        self._cfg, self._model, _ = infer_utils.load_xvc(
            str(config_path), str(self._root / "xvc" / "xvc.pt"), device_index, False
        )
        self._speaker: Any = None
        self._frame: Any = None

    def _write_runtime_config(self, source: Path) -> Path:
        """Point X-VC's yaml at the downloaded tokenizer and speaker encoder."""
        from omegaconf import OmegaConf  # type: ignore[import-not-found]

        cfg = OmegaConf.load(str(source / "configs" / "xvc.yaml"))
        body = cfg["config"] if "config" in cfg else cfg
        tokenizer = str(self._root / "glm-4-voice-tokenizer")
        generator = body["model"]["generator"]
        generator["semantic_encoder"]["encoder"]["from_pretrained"]["local_ckpt"] = (
            tokenizer
        )
        generator["semantic_encoder"]["cfg"]["local_ckpt"] = tokenizer
        generator["speaker_encoder"]["pretrained_dir"] = str(
            self._root / "speech_eres2net_sv_en_voxceleb_16k"
        )
        out = self._root / "xvc-runtime.yaml"
        OmegaConf.save(cfg, str(out))
        return out

    def _tensor(self, samples: np.ndarray) -> Any:
        array = np.asarray(samples, dtype=np.float32).reshape(-1)
        pad = (-array.size) % XVC_LATENT_HOP
        if pad:
            array = np.pad(array, (0, pad))
        return self._torch.from_numpy(array)[None, None, :].to(self._device)

    def set_reference(self, clip: np.ndarray, sample_rate: int) -> None:
        clip = _resample_linear(clip, sample_rate, self.sample_rate)
        target = self._tensor(clip)
        self._speaker, self._frame = self._infer.precompute_conditions(
            self._model, target, target
        )

    def convert_window(self, window: np.ndarray) -> np.ndarray:
        if self._speaker is None:
            raise RuntimeError("set_reference must be called before converting")
        out = self._infer.run_stream_chunk_forward(
            self._model, self._tensor(window), self._speaker, self._frame
        )
        audio = out.squeeze().detach().float().cpu().numpy()
        return np.asarray(audio, dtype=np.float32)[: window.size]


class QwenVoiceDesign:
    """``ReferenceClipMaker`` over Qwen3-TTS VoiceDesign via ``qwen_tts``."""

    def __init__(self, root: Path, *, device_index: int = 0) -> None:
        status = pack_status(VOICE_DESIGN_PACK, Path(root))
        if not status.installed:
            missing = ", ".join(item.relpath for item in status.missing)
            raise RuntimeError(f"VoiceDesign pack incomplete: missing {missing}")
        import torch
        from qwen_tts import Qwen3TTSModel  # type: ignore[import-not-found]

        _cuda_device(device_index)
        self._model = Qwen3TTSModel.from_pretrained(
            str(Path(root) / "qwen3-tts-voicedesign"),
            device_map=f"cuda:{device_index}",
            dtype=torch.bfloat16,
        )

    def make_clip(
        self, instruct: str, text: str = DESIGN_SAMPLE_TEXT
    ) -> tuple[np.ndarray, int]:
        wavs, rate = self._model.generate_voice_design(
            text=text, language="English", instruct=instruct
        )
        return np.asarray(wavs[0], dtype=np.float32).reshape(-1), int(rate)


def _resample_linear(samples: np.ndarray, rate_in: int, rate_out: int) -> np.ndarray:
    """Plain linear resampling for reference clips (quality is not critical)."""
    x = np.asarray(samples, dtype=np.float32).reshape(-1)
    if rate_in == rate_out or x.size == 0:
        return x
    n_out = int(round(x.size * rate_out / rate_in))
    positions = np.linspace(0.0, x.size - 1, n_out)
    return np.interp(positions, np.arange(x.size), x).astype(np.float32)

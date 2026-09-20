# Copyright (C) 2026 Michael Cobb
# SPDX-License-Identifier: GPL-3.0-or-later

"""Engine seam: block-wise processing shared by Preview and Roleplay Mode."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np

DEFAULT_BLOCK_SIZE = 256
DEFAULT_SAMPLE_RATE = 48_000


@dataclass(frozen=True)
class EngineCapabilities:
    changes_identity: bool
    requires_gpu: bool


@dataclass(frozen=True)
class ParameterSpec:
    key: str
    label: str
    minimum: float
    maximum: float
    default: float


@runtime_checkable
class Engine(Protocol):
    """Processing method that turns Dry Voice into Character Voice."""

    block_size: int
    sample_rate: int

    def process_block(self, block: np.ndarray) -> np.ndarray:
        """Process one float32 block of ``block_size`` samples."""

    def render(self, take: np.ndarray) -> np.ndarray:
        """Render a Take by looping ``process_block`` (Preview path)."""

    def parameter_schema(self) -> tuple[ParameterSpec, ...]:
        """Declare slider parameters for this Engine."""

    def capabilities(self) -> EngineCapabilities:
        """Capability flags: identity change and GPU requirement."""

    def set_params(self, params: dict[str, Any]) -> None:
        """Patch Engine parameters. Only listed keys are updated."""

    def apply_macro(self, tag: str) -> None:
        """Apply a Tone Tag macro as a partial parameter patch."""


class BaseEngine:
    """Default ``render`` that feeds a Take through ``process_block``."""

    block_size: int = DEFAULT_BLOCK_SIZE
    sample_rate: int = DEFAULT_SAMPLE_RATE

    def __init__(
        self,
        *,
        block_size: int = DEFAULT_BLOCK_SIZE,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        macros: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self.block_size = block_size
        self.sample_rate = sample_rate
        self._params: dict[str, Any] = {}
        self._macros: dict[str, dict[str, Any]] = macros or {}

    def process_block(self, block: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def render(self, take: np.ndarray) -> np.ndarray:
        samples = _as_mono_float32(take)
        if samples.size == 0:
            return samples.copy()
        padded = _pad_to_block_size(samples, self.block_size)
        chunks: list[np.ndarray] = []
        for start in range(0, padded.size, self.block_size):
            chunks.append(self.process_block(padded[start : start + self.block_size]))
        out = np.concatenate(chunks)
        return out[: samples.size]

    def parameter_schema(self) -> tuple[ParameterSpec, ...]:
        return ()

    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(changes_identity=False, requires_gpu=False)

    def set_params(self, params: dict[str, Any]) -> None:
        self._params.update(params)

    def apply_macro(self, tag: str) -> None:
        patch = self._macros.get(tag)
        if patch:
            self.set_params(patch)

    def params(self) -> dict[str, Any]:
        return dict(self._params)


class PassthroughEngine(BaseEngine):
    """Identity Engine for tests: Character Voice equals Dry Voice."""

    def process_block(self, block: np.ndarray) -> np.ndarray:
        samples = _as_mono_float32(block)
        if samples.size != self.block_size:
            raise ValueError(
                f"process_block expected {self.block_size} samples, got {samples.size}"
            )
        return samples.copy()


def _as_mono_float32(samples: np.ndarray) -> np.ndarray:
    array = np.asarray(samples, dtype=np.float32)
    if array.ndim != 1:
        raise ValueError(f"expected mono audio (1-D), got shape {array.shape}")
    return array


def _pad_to_block_size(samples: np.ndarray, block_size: int) -> np.ndarray:
    remainder = samples.size % block_size
    if remainder == 0:
        return samples
    padding = block_size - remainder
    return np.concatenate([samples, np.zeros(padding, dtype=np.float32)])

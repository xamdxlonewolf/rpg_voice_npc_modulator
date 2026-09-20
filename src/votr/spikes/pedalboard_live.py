# Copyright (C) 2026 Michael Cobb
# SPDX-License-Identifier: GPL-3.0-or-later

"""S0.5 harness: Pedalboard Reverb / Lowpass / Compressor block-wise."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass

import numpy as np

from votr.engine import BaseEngine

SAMPLE_RATE = 48_000


def pedalboard_available() -> bool:
    try:
        import pedalboard  # noqa: F401
    except ImportError:
        return False
    return True


class PedalboardEffectsEngine(BaseEngine):
    """Live-style effects: process each block with ``reset=False``."""

    def __init__(
        self,
        *,
        block_size: int = 256,
        sample_rate: int = SAMPLE_RATE,
        reset: bool = False,
    ) -> None:
        super().__init__(block_size=block_size, sample_rate=sample_rate)
        import pedalboard as pb

        self._board = pb.Pedalboard(
            [
                pb.Compressor(threshold_db=-16, ratio=4, attack_ms=2, release_ms=80),
                pb.LowpassFilter(cutoff_frequency_hz=1200),
                pb.Reverb(room_size=0.3, damping=0.4, wet_level=0.25, dry_level=0.7),
            ]
        )
        self.reset_each_block = reset

    def process_block(self, block: np.ndarray) -> np.ndarray:
        samples = np.asarray(block, dtype=np.float32)
        if samples.size != self.block_size:
            raise ValueError(
                f"process_block expected {self.block_size} samples, got {samples.size}"
            )
        return self._board.process(
            samples,
            self.sample_rate,
            buffer_size=self.block_size,
            reset=self.reset_each_block,
        )


def _sine(seconds: float, sample_rate: int, hz: float = 220.0) -> np.ndarray:
    t = np.arange(int(seconds * sample_rate), dtype=np.float32) / sample_rate
    return (0.2 * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def boundary_step_ratio(audio: np.ndarray, hop: int) -> float:
    bounds = []
    interiors = []
    for i in range(hop, audio.size - 1, hop):
        bounds.append(abs(audio[i] - audio[i - 1]))
        interiors.append(abs(audio[i - hop // 2] - audio[i - hop // 2 - 1]))
    return float(np.mean(bounds) / (np.mean(interiors) + 1e-12))


@dataclass(frozen=True)
class EffectMeasurement:
    name: str
    hop: int
    reset: bool
    rms_vs_offline: float
    boundary_ratio: float
    finite: bool
    peak: float


def collect_measurements() -> list[EffectMeasurement]:
    if not pedalboard_available():
        return []
    import pedalboard as pb

    sine = _sine(1.0, SAMPLE_RATE)
    rows: list[EffectMeasurement] = []
    plugins = {
        "Reverb": lambda: pb.Reverb(
            room_size=0.3, damping=0.4, wet_level=0.25, dry_level=0.7
        ),
        "LowpassFilter": lambda: pb.LowpassFilter(cutoff_frequency_hz=1200),
        "Compressor": lambda: pb.Compressor(
            threshold_db=-16, ratio=4, attack_ms=2, release_ms=80
        ),
    }
    for name, factory in plugins.items():
        for hop in (256, 512):
            for reset in (False, True):
                plugin = factory()
                offline = factory().process(sine, SAMPLE_RATE, reset=True)
                chunks = []
                for i in range(0, sine.size, hop):
                    chunks.append(
                        plugin.process(sine[i : i + hop], SAMPLE_RATE, reset=reset)
                    )
                streamed = np.concatenate(chunks)[: sine.size]
                err = streamed - offline[: streamed.size]
                rows.append(
                    EffectMeasurement(
                        name=name,
                        hop=hop,
                        reset=reset,
                        rms_vs_offline=float(np.sqrt(np.mean(err**2))),
                        boundary_ratio=boundary_step_ratio(streamed, hop),
                        finite=bool(np.isfinite(streamed).all()),
                        peak=float(np.max(np.abs(streamed))),
                    )
                )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    for row in collect_measurements():
        print(asdict(row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

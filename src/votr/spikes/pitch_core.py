# Copyright (C) 2026 Michael Cobb
# SPDX-License-Identifier: GPL-3.0-or-later

"""S0.3 harness: measure rubband LiveShifter vs python-stretch."""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict, dataclass

import numpy as np

from votr.engine import BaseEngine

SAMPLE_RATE = 48_000
CLICK_AT = 0.10  # seconds into the Take


@dataclass(frozen=True)
class ShifterMeasurement:
    name: str
    available: bool
    detail: str
    stream_block: int | None = None
    configure_ms: float | None = None
    native_block: int | None = None
    start_delay_ms: float | None = None
    reported_latency_ms: float | None = None
    cpu_percent: float | None = None
    windows_wheel: bool | None = None


def rubband_available() -> bool:
    try:
        import rubband  # noqa: F401
    except ImportError:
        return False
    return True


def stretch_available() -> bool:
    try:
        import python_stretch  # noqa: F401
    except ImportError:
        return False
    return True


class StretchEngine(BaseEngine):
    """python-stretch adapter. ``process`` is 1:1 when timeFactor is 1."""

    def __init__(
        self,
        *,
        block_size: int = 256,
        sample_rate: int = SAMPLE_RATE,
        configure_ms: float = 60.0,
        pitch_semitones: float = 0.0,
    ) -> None:
        super().__init__(block_size=block_size, sample_rate=sample_rate)
        import python_stretch as ps

        block = max(1, int(sample_rate * configure_ms / 1000.0))
        interval = max(1, block // 4)
        self._stretch = ps.Signalsmith.Stretch()
        self._stretch.configure(1, block, interval)
        self._stretch.setTimeFactor(1.0)
        self._stretch.setTransposeSemitones(pitch_semitones)
        self._in = np.zeros((1, block_size), dtype=np.float32)
        self.configure_ms = configure_ms
        self.native_block = self._stretch.blockSamples()

    def reported_latency_frames(self) -> int:
        return int(self._stretch.outputLatency())

    def process_block(self, block: np.ndarray) -> np.ndarray:
        samples = np.asarray(block, dtype=np.float32)
        if samples.size != self.block_size:
            raise ValueError(
                f"process_block expected {self.block_size} samples, got {samples.size}"
            )
        self._in[0] = samples
        return self._stretch.process(self._in)[0].copy()


class LiveShifterEngine(BaseEngine):
    """rubband LiveShifter adapter. Native block is typically 512 at 48 kHz."""

    def __init__(
        self,
        *,
        sample_rate: int = SAMPLE_RATE,
        pitch_semitones: float = 0.0,
        formant_scale: float = 1.0,
    ) -> None:
        import rubband

        options = rubband.LiveOptions(formant=rubband.LiveFormantOption.preserved)
        self._shifter = rubband.LiveShifter(sample_rate, 1, options=options)
        native = int(self._shifter.get_block_size())
        super().__init__(block_size=native, sample_rate=sample_rate)
        self._shifter.set_pitch_scale(2.0 ** (pitch_semitones / 12.0))
        self._shifter.set_formant_scale(formant_scale)
        self._out = np.zeros(native, dtype=np.float32)

    def reported_latency_frames(self) -> int:
        return int(self._shifter.get_start_delay())

    def process_block(self, block: np.ndarray) -> np.ndarray:
        samples = np.asarray(block, dtype=np.float32)
        if samples.size != self.block_size:
            raise ValueError(
                f"process_block expected {self.block_size} samples, got {samples.size}"
            )
        self._shifter.shift_into(samples, self._out)
        return self._out.copy()


def _click_take(seconds: float, sample_rate: int) -> tuple[np.ndarray, int]:
    n = int(seconds * sample_rate)
    take = np.zeros(n, dtype=np.float32)
    index = int(CLICK_AT * sample_rate)
    take[index] = 0.9
    return take, index


def measure_engine(engine: BaseEngine, *, seconds: float = 2.0) -> dict[str, float]:
    take, click_at = _click_take(seconds, engine.sample_rate)
    t0 = time.perf_counter()
    rendered = engine.render(take)
    elapsed = time.perf_counter() - t0
    peak = int(np.argmax(np.abs(rendered)))
    audio_s = take.size / engine.sample_rate
    return {
        "cpu_percent": elapsed / audio_s * 100.0,
        "start_delay_ms": (peak - click_at) / engine.sample_rate * 1000.0,
        "reported_latency_ms": engine.reported_latency_frames()
        / engine.sample_rate
        * 1000.0,
        "native_block": float(getattr(engine, "native_block", engine.block_size)),
    }


def collect_measurements() -> list[ShifterMeasurement]:
    rows: list[ShifterMeasurement] = []
    rows.append(
        ShifterMeasurement(
            name="rubband.LiveShifter",
            available=rubband_available(),
            detail=(
                "PyPI wheels: macosx_14_0_arm64 only (0.2.0–0.3.1). "
                "No win_amd64 or manylinux wheel. Source build needs Rubber Band 4.x."
            ),
            windows_wheel=False,
        )
    )
    if rubband_available():
        engine = LiveShifterEngine(pitch_semitones=5.0, formant_scale=1.0)
        stats = measure_engine(engine)
        rows.append(
            ShifterMeasurement(
                name="rubband.LiveShifter OptionFormantPreserved",
                available=True,
                detail=(
                    "set_formant_scale + shift_into; native block fixed by Rubber Band."
                ),
                stream_block=engine.block_size,
                native_block=engine.block_size,
                start_delay_ms=stats["start_delay_ms"],
                reported_latency_ms=stats["reported_latency_ms"],
                cpu_percent=stats["cpu_percent"],
                windows_wheel=False,
            )
        )
    rows.append(
        ShifterMeasurement(
            name="python-stretch",
            available=stretch_available(),
            detail=(
                "Wheels include win_amd64 and manylinux. "
                "0.3.1 exposes configure/setTransposeSemitones; no setFormantFactor."
            ),
            windows_wheel=True,
        )
    )
    if stretch_available():
        for ms in (60.0, 120.0):
            for hop in (256, 512):
                engine = StretchEngine(
                    block_size=hop, configure_ms=ms, pitch_semitones=5.0
                )
                stats = measure_engine(engine)
                rows.append(
                    ShifterMeasurement(
                        name=f"python-stretch configure {ms:.0f} ms",
                        available=True,
                        detail="timeFactor=1; process() returns one block per block.",
                        stream_block=hop,
                        configure_ms=ms,
                        native_block=engine.native_block,
                        start_delay_ms=stats["start_delay_ms"],
                        reported_latency_ms=stats["reported_latency_ms"],
                        cpu_percent=stats["cpu_percent"],
                        windows_wheel=True,
                    )
                )
    return rows


def format_report(rows: list[ShifterMeasurement]) -> str:
    lines = ["# S0.3 pitch/formant measurements", ""]
    for row in rows:
        lines.append(f"## {row.name}")
        for key, value in asdict(row).items():
            if key == "name":
                continue
            lines.append(f"- {key}: {value}")
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    print(format_report(collect_measurements()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

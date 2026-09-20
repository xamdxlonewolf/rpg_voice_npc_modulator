# Spike S0.4 — glass-to-glass latency through VB-CABLE

Date: 2026-09-20. **Blocked on Michael's Windows machine.**

## Planned harness

`votr.spikes.latency` finds `CABLE Input` / `CABLE Output` by device-name
pattern (ADR-0006: detect, do not bundle). It is meant to:

1. Play a click through the Engine into `CABLE Input`.
2. Capture at `CABLE Output`.
3. Measure the sample offset at block sizes 256 and 512, WASAPI shared and
   exclusive.

That harness becomes the E5 Latency Test.

## What this VM can do

- PortAudio is installed; **device list is empty** (ALSA and OSS have no
  devices).
- No VB-CABLE, no WASAPI.
- Unit tests cover name matching and click-offset math only.
- `python -m votr.spikes.latency` records four blocked rows (256/512 ×
  shared/exclusive). `measured_ms` is **None**.

## What is not measured (do not invent)

- Glass-to-glass milliseconds on Michael's PC.
- WASAPI shared vs exclusive.
- Whether 256 frames stays glitch-free through the cable.

Re-run this spike on Windows 10/11 with VB-CABLE installed after a reboot.
Until then the ~120 ms default stays a target, not a measurement.

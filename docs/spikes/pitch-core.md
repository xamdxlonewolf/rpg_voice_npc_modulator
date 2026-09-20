# Spike S0.3 — live pitch/formant core

Date: 2026-09-20. Machine: Linux cloud VM (x86_64, no audio devices).
Not Michael's Windows box. Do not treat these as glass-to-glass numbers.

## What we wired

- `votr.live.DuplexStream`: sounddevice duplex `Stream` at 48 kHz, float32,
  blocksize 256 or 512 (Engine-declared). Callback zeros `outdata`, copies the
  processed block only, uses a preallocated input buffer, and runs on PortAudio's
  I/O thread (`votr-audio` starter thread).
- This VM has PortAudio but **zero ALSA/OSS devices**, so the live stream cannot
  be opened here. Unit tests drive the callback without a device.

## Libraries

| Library | Install on this VM | Windows wheel on PyPI |
| --- | --- | --- |
| `rubband` 0.3.1 `LiveShifter` | Source build against Rubber Band **4.0.0** (Ubuntu ships 3.3 only). Published wheels are **macosx_14_0_arm64 only** for every 0.2.0–0.3.1 release. | **No** `win_amd64` wheel |
| `python-stretch` 0.3.1 | manylinux wheel | **Yes** (`cp312-abi3-win_amd64` and older CPython tags) |

`LiveShifter` API confirmed after the source build: `LiveOptions(formant=preserved)`
(`OptionFormantPreserved`), `set_formant_scale`, `get_block_size()` **512** at
48 kHz, `get_start_delay()`, `shift` / `shift_into`.

`python-stretch` 0.3.1 has `configure(nChannels, blockSamples, intervalSamples)`,
`setTransposeSemitones`, `setTimeFactor`, `process`. It does **not** expose
`setFormantFactor` / `setFormantSemitones` (C++ Signalsmith has them; the binding
does not).

## Offline measurements (2 s click Take, +5 semitones)

| Core | Stream hop | Native / configure | Reported delay | Click-to-peak | CPU % of realtime |
| --- | --- | --- | --- | --- | --- |
| rubband LiveShifter, formant preserved | 512 (forced) | block 512 | 50.2 ms (`get_start_delay`) | **51.0 ms** | 3.1 % |
| python-stretch | 256 | configure 60 ms (block 2880, interval 720) | 30.0 ms (`outputLatency`) | 0.0 ms (1:1 aligned) | 2.4 % |
| python-stretch | 512 | configure 60 ms | 30.0 ms | 0.0 ms | 1.2 % |
| python-stretch | 256 | configure 120 ms (block 5760, interval 1440) | 60.0 ms | 0.0 ms | 4.8 % |
| python-stretch | 512 | configure 120 ms | 60.0 ms | 0.0 ms | 2.5 % |

Click-to-peak on Stretch is not a substitute for `outputLatency`: `process()` is
1:1 at `timeFactor=1`, so the impulse peak stays aligned. Use the reported
30 / 60 ms as the algorithmic delay. LiveShifter's click-to-peak matches its
start delay (~50 ms).

## Pick

- **Primary (quality / live API): `rubband.LiveShifter`** with
  `OptionFormantPreserved` and `setFormantScale`. ~50 ms start delay, ~3 % CPU,
  independent formants, fixed 512-frame block. **Cannot `pip install` on Windows
  today.** S0.6 must vendor a Windows wheel or we cannot ship this as the
  installer default.
- **Fallback / current Windows-installable core: `python-stretch`** at
  `configure()` **60 ms** (interval 15 ms). ~30 ms reported delay, ~1–2.5 % CPU,
  win_amd64 wheel. Pitch only until the binding grows formant setters.

Rough-draft plan: implement S1.2 against **python-stretch 60 ms** so a Windows
`pip`/installer path works; keep the LiveShifter adapter and switch primary if
S0.6 produces a Windows `rubband` wheel.

## Not measured

- Start delay and CPU **on Windows**.
- A real duplex callback under load (no devices on this VM).
- Glass-to-glass (that is S0.4, needs VB-CABLE on Michael's machine).

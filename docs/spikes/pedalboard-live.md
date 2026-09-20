# Spike S0.5 — Pedalboard block-wise live effects

Date: 2026-09-20. Machine: Linux cloud VM. Offline block processing only
(no audio devices). Pedalboard 0.9.25.

## Question

Are Pedalboard `Reverb`, `LowpassFilter`, and `Compressor` glitch-free when
called every 256/512 frames with `reset=False`, or do we need NumPy/SciPy
equivalents?

## Result

**Use Pedalboard with `reset=False`.** Do not reset per callback.

On a 1 s 220 Hz sine at 48 kHz:

| Plugin | Hop | `reset=False` RMS vs one-shot | Boundary/interior step | `reset=True` boundary ratio |
| --- | --- | --- | --- | --- |
| Reverb | 256 / 512 | **0** | ~1.0 | ~1.0 (error vs offline, not a click) |
| LowpassFilter | 256 / 512 | **0** | ~1.0 | **~32** (clear clicks) |
| Compressor | 256 / 512 | **0** | ~1.0 | **~5** (clicks) |

`reset=False` streamed audio is bit-identical to a single `process()` of the
whole buffer. Boundary sample steps match the interior of the sine. Output is
finite; peaks stay bounded.

`reset=True` on LowpassFilter and Compressor restarts filter/envelope state
every block and produces audible discontinuities. Never do that in the live
callback.

## Pick

Ship Pedalboard for live reverb / low-pass / compressor. A `Pedalboard` chain
with `buffer_size=block_size` and `reset=False` is the live path. NumPy/SciPy
equivalents are not required for these three.

Caveats: `process()` still allocates a return array (same class of issue as
python-stretch). Not measured inside a real PortAudio callback (no devices).
Not measured on Windows.

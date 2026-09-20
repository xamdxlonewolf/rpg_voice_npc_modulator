# Voice of the Realm — Rough-Draft Backlog

Scope: the rough draft where Michael can **create a Voice, save it as a Voice Button,
Preview it (record a Take, hear it through the Voice on his speakers), and enter Roleplay
Mode so only the Character Voice reaches Discord through a Virtual Cable**. Decisions
are in `docs/adr/0001`–`0006` and `docs/project-context.md`; vocabulary in
`docs/CONTEXT.md`. Epic 7 (Neural Engine) is **not** in the rough draft.

Conventions: `E#` epic, `S#.#` story (user-facing, has acceptance criteria), tasks are
the bullets under each story. Sizes are relative: S = one sitting, M = a few, L = many.
Order within an epic is the suggested build order.

Rough-draft definition of done: on a clean Windows 10/11 machine with VB-CABLE installed,
Michael installs the app, passes first-run setup, creates a Voice named "Grimjaw" with the
tags `gravelly` and `booming`, saves it, records a Take and hears it as Grimjaw on his
speakers, clicks the Grimjaw Voice Button, enters Roleplay Mode, and a Player in Discord
hears only Grimjaw.

---

## E0 — Foundation and spikes

Goal: prove the audio path and lock the seams before UI work.

**S0.1 — Repo scaffold (S).** As a developer I can clone, install, run tests, and launch
an empty PySide6 window.
- `pyproject.toml` (Python 3.11+), `src/votr/` package layout, `ruff`, `pytest`.
- GPL-3.0 `LICENSE`, `THIRD_PARTY_NOTICES.md` skeleton (Rubber Band, FFTW, JUCE via
  Pedalboard, PortAudio, Qt/PySide6 LGPL).
- `python -m votr` opens a window titled "Voice of the Realm".
- Note: repo writes need Michael's go-ahead; until then this lands as a plan only.

**S0.2 — Engine interface (S).** As a developer I can implement an Engine without touching
UI or storage.
- `Engine` protocol: `block_size`, `sample_rate`, `process_block(in) -> out`,
  `render(take) -> audio` (default impl loops `process_block`), `parameter_schema()`,
  `capabilities()` (`changes_identity`, `requires_gpu`), `set_params(dict)`,
  `apply_macro(tag)`.
- `PassthroughEngine` for tests (identity).
- Unit tests: `render` on a known WAV equals concatenated `process_block` output.

**S0.3 — Spike: live pitch/formant core (M).** As a developer I know which shifter we ship.
- Wire `sounddevice` duplex `Stream` (48 kHz, float32, `blocksize` 256 and 512) with an
  allocation-free callback on a dedicated thread.
- Try `rubband` `LiveShifter` (`OptionFormantPreserved`, `setFormantScale`); confirm the
  Windows wheel exposes it; measure start delay and CPU %.
- Try `python-stretch` with `configure()` at 60 ms and 120 ms blocks; same measurements.
- Record results in `docs/spikes/pitch-core.md`; pick primary and fallback.

**S0.4 — Spike: glass-to-glass latency through VB-CABLE (S).** As a developer I have a
real latency number on Michael's machine.
- Loopback test harness: play a click into the engine, capture at `CABLE Output`,
  measure offset; repeat at 256/512 frames, WASAPI shared vs exclusive.
- Record in `docs/spikes/latency.md`. This harness becomes the Latency Test (E5).

**S0.5 — Spike: Pedalboard block-wise effects live (S).** As a developer I know whether
Pedalboard `Reverb`/`LowpassFilter`/`Compressor` with `reset=False` are glitch-free in the
callback, or whether we use NumPy/SciPy equivalents.

**S0.6 — Spike: PyInstaller bundle (S).** As a developer I know the bundle builds with
`rubband` + `pedalboard` + PySide6 on Windows, its size, and cold-start time.

---

## E1 — Create a Voice

Goal: a GM can define a character sound with a name, Tone Hints, Tone Tags, and sliders.

**S1.1 — Voice model and JSON storage (S).** As a GM my Voices persist between sessions.
- `Voice` dataclass: `id`, `name`, `engine_id`, `tone_hints` (free text), `tone_tags`
  (list), `params` (dict per Engine schema), `colour`/`icon`, `created`, `updated`.
- One JSON per Voice in `platformdirs.user_data_dir("VoiceOfTheRealm")/voices/`.
- Load all on start; tolerate a corrupt file (skip, log, warn).
- Tests: round-trip, unknown fields preserved (forward compatibility).

**S1.2 — DSP Engine v1 (M).** As a GM I can shape pitch, body, and texture.
- Parameters: `pitch_semitones` (−12..+12), `formant_semitones` (−12..+12),
  `tonality`/timbre preservation, `growl` (saturation), `hollow` (ring-mod/comb mix),
  `room` (short reverb), `distance` (low-pass + level), `breath` (noise layer), `gate`.
- Chain: gate → LiveShifter (pitch, formant) → effects → limiter; fixed block size.
- `render()` used by Preview; `process_block()` by Roleplay Mode; same chain.
- Tests: silence in → silence out; RMS bounded; latency reported via `latency_frames()`.

**S1.3 — Tone Tags and Macros (S).** As a GM I can type "gravelly" and get sensible
sliders.
- `macros.json` bundled: ~12 tags (`gravelly`, `booming`, `frail`, `hollow`, `tiny`,
  `giant`, `ghostly`, `robotic`, `nasal`, `whisper`, `regal`, `sly`) each with a
  partial `params` patch and a one-line description.
- Applying a tag patches only the keys it defines; later tags win; GM edits after that
  are kept until the tag is re-applied.
- Tests: apply order, idempotence.

**S1.4 — Voice editor screen (M).** As a GM I can create and edit a Voice.
- Fields: name, colour, Tone Hints (multi-line), Tone Tags (chips with autocomplete
  from `macros.json`), sliders generated from `parameter_schema()`.
- "Save", "Save as new", "Delete" with confirm; unsaved-changes guard.
- Preview panel embedded (E3) so tuning happens here.
- Free-text Tone Hints are stored verbatim and shown on the Voice Button tooltip.

---

## E2 — Voice Library and Voice Buttons

Goal: saved Voices are one click away; the Active Voice is obvious.

**S2.1 — Library grid (S).** As a GM I see all my Voices as buttons.
- Grid of Voice Buttons (name, colour, first two Tone Tags); "New Voice" tile.
- Sort by name / last used; search box filtering by name or tag.
- Empty state explains "create your first Voice".

**S2.2 — Select the Active Voice (S).** As a GM clicking a button makes it the Active
Voice everywhere.
- Single selection, highlighted; Active Voice name shown in the header and in
  Roleplay Mode panel.
- Switching Voices while Roleplay Mode is on swaps Engine params at a block boundary
  with a 20–50 ms crossfade to avoid clicks.
- Voices whose `engine_id` is not installed are visible but disabled with a tooltip.

**S2.3 — Button actions (S).** Right-click / long-press: Edit, Duplicate, Delete,
Reveal file.

---

## E3 — Delayed Preview (record a Take, play it through the Voice)

Goal: the GM hears exactly what Players will hear, on his own speakers, while tuning.

**S3.1 — Record a Take (S).** As a GM I hold a button, speak, and release.
- Push-to-talk record button (also spacebar while the editor is focused); max 15 s;
  level meter and countdown while recording.
- Take stored in memory and as `takes/last.wav`; "Keep this Take" pins it as
  `takes/<name>.wav` for reuse.
- Bundled sample phrase (`assets/sample_take.wav`, a neutral GM line) used when no mic
  is present or before the first recording.

**S3.2 — Render and play through the Active Voice (S).** As a GM I hear the Take as the
Voice on my speakers within a moment of releasing the button.
- `engine.render(take)` on a worker thread; play via `sounddevice` to the **speakers**
  Output Device (never the Virtual Cable).
- Auto-play on release; "Play again" button; stop button.
- Render must go through `process_block` (ADR-0005). Test: render output equals a live
  pass of the same audio through the streaming path.

**S3.3 — Tune-and-replay loop (S).** As a GM moving a slider replays the Take with the
new settings.
- Debounced (≈300 ms after the last change) re-render + replay; toggle to disable
  auto-replay.
- A/B: "Compare with dry" plays the unprocessed Take.

**S3.4 — Preview any Voice from the Library (S).** As a GM hovering/right-clicking a Voice
Button lets me play the last Take through that Voice without making it Active.

---

## E4 — Roleplay Mode into Discord via Virtual Cable

Goal: one switch; only the Character Voice reaches Discord; the GM can trust it.

**S4.1 — Duplex live stream (M).** As a GM I flip Roleplay Mode on and my mic is processed
live into the Virtual Cable.
- `sounddevice.Stream(device=(mic, cable_input))`, block size from the Latency Test.
- Callback: read block → `engine.process_block` → write. The dry buffer is never copied
  to `outdata`; a unit test asserts that with `PassthroughEngine` replaced by a
  "silence" engine, output is silent.
- Underrun/overrun counters surfaced in the UI; auto-recover on device loss.

**S4.2 — Roleplay Mode panel (S).** As a GM I can see it is live and working.
- Big On/Off toggle; Active Voice name; input meter (mic) and output meter (cable).
- "Discord is receiving" hint appears when the output meter shows signal.
- Cannot turn on without a Virtual Cable configured (points to the wizard).

**S4.3 — Panic/mute hotkey (S).** As a GM I can cut the output instantly.
- Global hotkey (default `Ctrl+Shift+M`) mutes the cable output (writes silence, stream
  keeps running); visible red state; second press unmutes.
- Optional "hold to talk" mode (mute unless held).

**S4.4 — Optional Monitor (S).** As a GM I can hear the Character Voice in my headphones.
- Toggle, off by default; second output stream to the speakers/headphones device fed
  from the same processed blocks via a lock-free ring buffer.
- Warning shown if the Monitor device is speakers (feedback into the mic).

**S4.5 — Discord setup wizard (M).** As a GM I get from "installed the app" to "Discord
hears my Voice" without guessing.
- Step 1: detect `CABLE Input`/`CABLE Output` by device-name pattern; if missing, link to
  VB-CABLE download with install + reboot instructions (ADR-0006), "Check again" button.
- Step 2: choose mic; choose speakers (for Preview/Monitor); cable is chosen
  automatically.
- Step 3: instruct "In Discord → Settings → Voice & Video → Input Device → CABLE Output";
  "Send test tone/Take to cable" button so the GM can watch Discord's input meter.
- Step 4: summary; re-runnable from Settings. Linux path creates a null sink via
  `pactl`; macOS path links to BlackHole (not in the Windows rough draft installer).

---

## E5 — First-run setup and Latency Test

Goal: sensible defaults per machine; default ~120 ms; manual override.

**S5.1 — First-run flow (S).** On first launch: welcome → device pick → Latency Test →
Discord wizard (E4.5) → Library. Skippable, re-runnable from Settings.

**S5.2 — Latency Test (M).** As a GM the app picks settings for my machine.
- Reuse the S0.4 harness: loop a click through mic → engine → cable → capture, or, when
  the cable cannot be captured, through the engine only plus device-reported latency.
- Try block sizes 256/512/1024 and shifter block settings; choose the smallest that
  runs 10 s with zero underruns and total latency ≤ ~120 ms; else the smallest
  glitch-free option; report the measured ms.
- Settings: "Latency" slider (Lower latency ↔ Cleaner sound) that overrides the
  auto-tuned values; "Re-run test".

**S5.3 — Settings screen (S).** Devices, Latency, hotkeys, Monitor default, data folder
location, "Open Voices folder", About (version, licences).

---

## E6 — Install and distribution (Windows)

Goal: a downloadable installer that works on a clean Windows 10/11 machine.

**S6.1 — PyInstaller `--onedir` build (M).** Spec file collecting `rubband`, `pedalboard`,
`sounddevice` (PortAudio DLL), PySide6 plugins, `assets/`, `macros.json`; runtime hooks
as needed; smoke test script that launches and exits.

**S6.2 — Inno Setup installer (S).** Per-user install (no admin), Start Menu shortcut,
uninstaller, version stamping from `pyproject.toml`; installer shows GPL and links to
`THIRD_PARTY_NOTICES.md`.

**S6.3 — Release pipeline (S).** GitHub Actions on tag: build, package, attach installer
to a GitHub Release with SHA-256; CHANGELOG entry.

**S6.4 — Clean-machine acceptance run (S).** Execute the rough-draft definition of done on
a fresh Windows VM; record results in `docs/spikes/acceptance-rough-draft.md`.

---

## E7 — Neural Engine (LATER / OPTIONAL — not in the rough draft)

Goal: "sound like a different person" for GMs with an NVIDIA GPU, behind the same Engine
interface, with Voices designed from free-text Tone Hints.

**S7.1 — GPU detection and Engine pack download (M).** Detect NVIDIA + VRAM; offer
"Install Neural Engine"; download torch/CUDA runtime and model weights to the user-data
dir with resume and checksum; hide the option on non-NVIDIA machines.

**S7.2 — Zero-shot Neural Engine (L).** Implement `Engine` over X-VC (MIT) streaming
inference: reference-clip registration, chunked `process_block`, 300–500 ms expected
latency, `capabilities.changes_identity = True`, `requires_gpu = True`. Fallback candidate:
Seed-VC (GPL-3.0, archived).

**S7.3 — Voice design from Tone Hints (L).** Use Qwen3-TTS VoiceDesign (Apache-2.0) to
turn the free-text Tone Hints into a 5–10 s reference clip; store it with the Voice; the
clip is also the Preview sample. Re-generate button; pick-from-several.

**S7.4 — Mixed Library (S).** Voice Buttons show an Engine badge; Voices for an
uninstalled Engine are disabled with a tooltip (already handled by S2.2); Neural Voices
warn about latency when Roleplay Mode is turned on.

**S7.5 — Future spike: CPU voice conversion via LLVC (S).** Evaluate shipping a few
pre-trained fantasy voices as an any-to-one CPU Engine (<20 ms). Research only.

---

## Suggested order

E0 (S0.1–S0.4 first) → E1 (S1.1–S1.2) → E3 (S3.1–S3.2) → E1 (S1.3–S1.4) → E2 → E4
(S4.1, S4.2, S4.5, S4.3) → E5 → E6 → remaining E3/E4 stories → acceptance run (S6.4).
E7 is scheduled only after the rough draft is accepted and Michael's GPU is known.

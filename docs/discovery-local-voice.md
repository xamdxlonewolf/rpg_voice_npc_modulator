# Discovery — Local, Free Character Voices for a GM

Research date: 2026-09-20. Sources are linked inline; latency numbers are the projects'
own published figures or user-reported measurements, not our benchmarks yet.

> **Updated after grill round 1.** Decisions now locked (see `docs/adr/`): online-only
> Roleplay Mode via a Virtual Cable (ADR-0002), GPL licence (ADR-0003), DSP Engine first
> (ADR-0004), delayed record-then-play Preview (ADR-0005), link-don't-bundle cables
> (ADR-0006). Sections 3, 5 and 7 were revised accordingly; in-person routing is kept only
> as a rejected option.

## 1. Short answer

- **Feasible, free, local, CPU-only, on any laptop:** *voice modulation* of the GM's own
  speech — pitch shift, formant shift, EQ, saturation, reverb, ring-mod, etc. The GM
  still does the acting; the app changes the timbre. ~100–150 ms glass-to-glass in
  Python. No AI models needed. This covers goblins, giants, ghosts, robots, old crones,
  "same actor, different mask".
- **Feasible, free, local, but GPU-gated:** *voice conversion* — making the GM sound like
  a genuinely different person (a female elf, a child, a specific timbre) — requires a
  neural model. Open, permissively licensed models exist (X-VC, MIT; RVC, MIT code) but
  real-time use needs an **NVIDIA GPU** (roughly 6 GB VRAM for zero-shot models) and
  lands at **300–500 ms** latency. On CPU these models run at ~1–2 s delay, which is
  unusable live.
- **"Describe the voice in words and get that voice"** is possible offline via local
  voice-design TTS (Qwen3-TTS VoiceDesign, Apache-2.0) feeding a zero-shot converter, but
  it is GPU-only and heavy (multi-GB downloads). In the DSP path, personality/tone text
  can only act as notes plus *tag → parameter macros* (e.g. "gravelly", "booming",
  "small"). It cannot invent a new timbre by itself. Being honest: without a GPU the
  "personality" text is metadata, not a synthesis input.
- **Roleplay Mode** ("GM does not hear their own dry voice") is trivial in software (we
  never write the dry signal to the output). Michael chose **online-only** play
  (ADR-0002), so the processed voice goes into Discord/etc. via a virtual cable and the
  in-person caveat (players hearing the real voice in the room, speaker feedback) is out
  of scope.

## 2. Approaches compared

| # | Approach | What it does | Hardware | Latency (algorithmic + I/O) | Licence | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| A | **DSP modulation** (pitch/formant + effects) | Alters the GM's timbre; identity mostly preserved | Any CPU | ~60–150 ms depending on block size | Rubber Band GPL (primary, app is GPL) or Signalsmith MIT (fallback) | **Rough-draft engine (ADR-0004).** Always works. |
| B | **Zero-shot neural VC** (X-VC, Seed-VC) | Sounds like a target reference clip; no training per voice | NVIDIA GPU, ~3–6 GB VRAM | 290–490 ms end-to-end reported | X-VC MIT; Seed-VC GPL-3.0 (archived) — both GPL-compatible | Later optional Neural Engine. Latency is fine for online play. |
| C | **Trained per-voice VC** (RVC via w-okada) | High quality per character, but needs a trained `.pth` per voice | NVIDIA GPU for real time (170–400 ms); CPU ≈ 1–2 s | RVC code MIT; embedder (ContentVec) licence unclear | Too much friction for "create a voice in the app". |
| D | **Distilled CPU VC** (LLVC, any-to-one) | ~20 ms on CPU, but one model per target voice; training needs an RVC teacher + GPU hours | CPU at runtime, GPU to train | <20 ms + I/O | MIT | Future "premium preset packs", not user-created voices. |
| E | **STT → LLM → TTS** re-speak | Recognise words, re-synthesise in a designed voice | GPU realistic | 1–3 s, loses the GM's acting/prosody | mixed | Rejected for live play. Fine for Preview of a designed voice. |
| F | Cloud voice APIs | — | — | — | paid | Rejected by ADR-0001. |

### A. DSP modulation — details

- **Signalsmith Stretch** ([C++ header-only, MIT](https://github.com/Signalsmith-Audio/signalsmith-stretch); Python wheels as
  [`python-stretch`](https://pypi.org/project/python-stretch/), MIT). Polyphonic pitch shift with independent
  **formant shift/compensation** (`setFormantFactor`, `setFormantBase`, `setTransposeSemitones`) and a
  tonality limit that preserves timbre. Default preset uses a 120 ms block / 30 ms interval, so
  algorithmic latency is ≈120 ms; `configure()` accepts smaller blocks (e.g. 40–60 ms) for lower latency
  at some quality cost. Runs at real time on modest CPUs. **Fallback** core now that the app is GPL.
- **Rubber Band 4 `LiveShifter`** ([GPL/commercial](https://breakfastquay.com/rubberband/code-doc/classRubberBand_1_1RubberBandLiveShifter.html); Python via
  [`rubband`](https://pypi.org/project/rubband/), wheels bundle Rubber Band 4.x): purpose-built live pitch
  shifter with `OptionFormantPreserved` and independent `setFormantScale`, "50 ms or more" delay, fixed
  block size from `get_block_size()`, allocation-free processing path. **Primary** live pitch/formant core
  under ADR-0003 (GPL app). Spike: confirm the `rubband` wheel ships `LiveShifter` on Windows and measure
  its start delay at 48 kHz.
- **Spotify Pedalboard** (GPLv3; `PitchShift` wraps Rubber Band): users report 1–2 s latency with
  `PitchShift` in `AudioStream`, and a 2026 PR is still fixing streaming silence ([#350](https://github.com/spotify/pedalboard/issues/350),
  [#486](https://github.com/spotify/pedalboard/pull/486)). Now allowed (GPL app) for the effects stages
  (reverb, EQ, distortion, compressor) applied block-wise after the pitch stage, and for Preview rendering;
  not for the live pitch path.
- **WORLD vocoder** (`pyworld`, MIT): great F0/formant control but designed for offline analysis/synthesis;
  poor fit for streaming.
- Everything else (EQ, saturation, ring modulation, chorus, short reverb, noise/breath layer, gate) is a few
  lines of NumPy/SciPy or small pure-Python DSP; no licence issue.
- Audio I/O: [`sounddevice`](https://github.com/spatialaudio/python-sounddevice) (PortAudio, MIT). Duplex
  `Stream` callback, `blocksize` 256–512 at 48 kHz, `latency='low'`. WASAPI on Windows, CoreAudio on macOS,
  ALSA/PipeWire on Linux. Callback must be allocation-free; pre-allocate buffers and keep the effect chain in
  vectorised NumPy or the C++ wrapper.

What DSP can and cannot do:

- Can: deeper/higher voice, bigger/smaller body (formants), age-ish cues, monster/ghost/robot/demon layers,
  "through a helmet", distance/room, whisper-to-rasp.
- Cannot: change gender convincingly for large shifts (chipmunk/robot artefacts beyond ±5–7 semitones without
  formant work; ±3–5 with), change accent or speaking style, or produce a specific "person". The GM's acting
  carries the personality.

### B. Zero-shot neural VC — details

- **X-VC** ([code MIT, weights on HF](https://github.com/Jerrister/X-VC), Apr 2026): one-step conversion in codec
  latent space; streaming with ~240 ms model latency, 2.4 s context window. A community client
  ([xvc-front](https://github.com/syedfahimabrar/xvc-front)) reports 290–490 ms end-to-end and states the model
  **cannot run on CPU in real time**; ~3 GB VRAM. Best current free zero-shot option by licence.
- **Seed-VC** ([GPL-3.0, repo archived](https://github.com/Plachtaa/seed-vc)): ~300 ms algorithmic + ~100 ms
  device delay on an RTX 3060 laptop GPU; real-time GUI included. Mature but GPL and unmaintained.
- **Beatrice v2** (used inside w-okada): low-latency CPU-capable, but **commercial use prohibited and
  redistribution needs Project Beatrice's permission** — excluded.
- **StreamVC / RT-VC** (Google etc.): closed source — excluded.
- **Zero-VC** (2026 paper, 20 ms zero-lookahead): research code; watch, don't depend on.

Designed-voice pipeline for the AI tier: personality text → **Qwen3-TTS-12Hz-1.7B-VoiceDesign**
([Apache-2.0](https://github.com/QwenLM/Qwen3-TTS), GPU) generates a 5–10 s reference clip → that clip is the
target for X-VC. Preview = play the clip; Roleplay = live conversion. Smaller CPU TTS options for Preview
only: **Chatterbox-Nano** (MIT, 110 M, 3× real time on 8 cores), **MOSS-TTS-Nano** (Apache-2.0, 0.1 B, ONNX
CPU), **Kokoro** (Apache-2.0, no cloning). None of these help the *live* path on CPU.

### C/D. RVC and LLVC — why not first

- RVC via [w-okada VCClient](https://github.com/w-okada/voice-changer) is the de-facto hobbyist real-time
  changer. Measured: ~170 ms (ONNX CUDA) vs ~1,060–1,820 ms (ONNX CPU). Needs a trained model per voice
  (minutes of clean target audio, GPU training). The default embedder shipped as `hubert_base.pt` is actually
  ContentVec with no clear licence on the HF page — a redistribution risk.
- [LLVC](https://github.com/KoeAI/LLVC) (MIT) proves <20 ms CPU voice conversion is possible, but each target
  voice is a separately trained model distilled from an RVC teacher. Attractive later as "ship 5–10
  pre-trained fantasy voices", not for user-created voices.

## 3. Roleplay Mode and audio routing (ADR-0002, ADR-0005, ADR-0006)

| Mode | Route | Notes |
| --- | --- | --- |
| **Roleplay Mode (online, the only live target)** | Mic → app → **Virtual Cable** → Discord/Roll20/Foundry picks the cable as its microphone | Dry path never written. Latency up to ~400 ms tolerable in conversation; we still default to ~120 ms. Optional Monitor: Character Voice also to the GM's headphones. |
| **Preview (delayed)** | Mic → record Take → render through the same streaming Engine path → **GM's speakers** | No feedback risk because the mic is only open while recording. Same block-wise processing as live, so Preview == what Players hear. |
| ~~In-person table~~ | Mic → app → room speaker | **Rejected (ADR-0002).** Players would hear the real voice too; feedback; hard latency ceiling. |

Virtual Cable options (all free, all a separate install; the app cannot silently create a Windows audio
device without a driver — hence ADR-0006, detect and link, never bundle):

- Windows: [VB-CABLE](https://vb-audio.com/Cable/) (donationware, ~1.3 MB, admin install, reboot required;
  volume/pro use needs a licence). Device names: `CABLE Input (VB-Audio Virtual Cable)` = app output,
  `CABLE Output (VB-Audio Virtual Cable)` = Discord input.
- macOS: [BlackHole 2ch](https://github.com/ExistentialAudio/BlackHole) (GPL-3.0; compatible with our GPL
  licence but still not bundled).
- Linux: PipeWire/PulseAudio `module-null-sink` + `module-remap-source` (no install; wizard can run `pactl`).

Two Output Devices are therefore in play at once: the Virtual Cable (Roleplay) and the GM's speakers
(Preview, Monitor). The app must keep both configured and never swap them. Roleplay Mode is: duplex
stream running with input = mic, output = cable, dry path never written to `outdata`, a big on/off
state, a level meter on the cable output so the GM can see Discord is receiving signal, and a global
panic/mute hotkey. The Discord wizard checks the cable exists, tells the GM to select `CABLE Output` in
Discord's Voice settings, and lets them run a Take-through-cable test.

## 4. Latency budget (DSP path, 48 kHz)

| Stage | Typical |
| --- | --- |
| Input device buffer (WASAPI shared) | 10–30 ms |
| Processing block (256–512 frames) | 5–11 ms |
| Pitch/formant stage: Rubber Band `LiveShifter` (≈50+ ms) or Signalsmith Stretch (block 60–120 ms) | 50–120 ms |
| Output buffer (Virtual Cable) | 10–30 ms |
| **Total** | **~75–190 ms**; target ≤120 ms (Q6 default), acceptable up to ~400 ms online |

Python is fine here as long as the audio callback does no allocation and no Python-level per-sample work.
Measure early with a loopback test (play click → record) — this is the first spike.

## 5. Stack for the rough draft (locked by grill round 1; ADR-0003, ADR-0004)

| Layer | Choice | Why | Alternative |
| --- | --- | --- | --- |
| Language | Python 3.11/3.12 | Confirmed | — |
| Licence | **GPL** (ADR-0003) | Opens Rubber Band and Pedalboard for the live path | — |
| Audio I/O | `sounddevice` (PortAudio, MIT) | Duplex callback, device enumeration, cross-platform | `pyaudio` |
| Pitch/formant | **`rubband` → Rubber Band 4 `LiveShifter`** (GPL) | Lowest-latency live shifter with formant preservation; fixed-block API suits the callback | `python-stretch` (Signalsmith, MIT) as fallback if the Windows wheel or start delay disappoints |
| Effects | `pedalboard` (GPLv3) block-wise for reverb/EQ/compressor/distortion + NumPy/SciPy for ring-mod, noise layer, gate | Ready-made, tested effects; GPL now allowed | Pure NumPy/SciPy |
| GUI | **PySide6 (Qt, LGPL)** | Confirmed; dynamic linking keeps LGPL terms satisfied | — |
| Voice storage | One JSON per Voice in the OS user-data dir (`platformdirs`); Takes as WAV alongside | Confirmed; import/export = copy a file | SQLite later |
| Packaging | PyInstaller `--onedir` + Inno Setup (Windows) | Confirmed; DSP-only bundle ≈ 80–150 MB with Pedalboard | Nuitka |
| Neural Engine (later, optional) | X-VC + Qwen3-TTS VoiceDesign via PyTorch CUDA, downloaded on demand | Base installer stays small; CUDA torch alone is ~2–2.5 GB | — |

Engine interface (the seam from ADR-0004): `process_block(in: float32[N]) -> float32[N]` with a fixed
block size declared by the Engine, `render(take: float32[...]) -> float32[...]` implemented by pushing
the Take through `process_block` (so Preview == live), `parameter_schema()`, `capabilities()`
(`changes_identity`, `requires_gpu`), and `apply_macro(tag)`.

## 6. Install story

1. Download the Windows installer (~100–150 MB). Run. No admin rights needed by the app itself; the
   Virtual Cable driver is the GM's own admin install (ADR-0006).
2. First run: pick mic and speakers; run the built-in Latency Test; run the "set up Discord" wizard, which
   detects VB-CABLE, links to the download if absent, and verifies `CABLE Input/Output` appear.
3. Later, optional (GPU owners): "Install Neural Engine" downloads torch/CUDA runtime + model weights
   (multi-GB) to the user-data dir. Detect NVIDIA via driver query; hide the option otherwise.

Packaging risks: PyInstaller and torch/onnxruntime provider DLLs need explicit collection (`collect_all`,
`--add-binary` for `onnxruntime_providers_*.dll`); ship CUDA payloads as a separate lazy download, as
[voicebox](https://github.com/jamiepine/voicebox) and OmniVoice-Studio do (CPU bundle ≈ 400–500 MB with
torch, ≈ 2.4 GB with CUDA).

## 7. Risks and honest limits

- **Expectation gap.** "Create a character voice from a personality description" reads like ElevenLabs
  voice design. Locally and free, that is GPU-only. On CPU the app changes *how the GM sounds*, not *who*.
  This must be explicit in the UI and in the epics.
- **Latency vs quality is a slider, not a switch.** Lower block sizes = more artefacts. Ship a latency test
  and per-machine tuning.
- **Virtual Cable is an external prerequisite.** If the GM has not installed VB-CABLE, Roleplay Mode
  cannot work; the wizard and startup checks must make this impossible to miss (ADR-0006).
- **Python audio callbacks** are not hard-real-time; glitches under GC/UI load are possible. Mitigate with
  larger blocks, a dedicated processing thread, and keeping Qt work off the audio thread. If it fails
  measurably, move the hot path to a small C++/Rust extension while keeping the app Python.
- **Licence hygiene (GPL app).** All bundled code must be GPL-compatible; ship licence texts and third-party
  notices. Beatrice (non-commercial) and the unclear-licence RVC embedder stay excluded. Link to, don't
  redistribute, VB-CABLE/BlackHole.
- **Preview fidelity.** If Preview ever renders through a different code path than Roleplay Mode, the GM
  will tune a Voice that sounds different to Players. Render Takes through `process_block` only (ADR-0005).
- **Model churn.** X-VC is five months old; Seed-VC is archived. The Engine abstraction protects against this.

## 8. Remaining unknowns (facts, resolved by spikes in `docs/backlog.md` Epic 0)

All round-1 decisions are settled (`docs/grill-round-1.md`). What is left is measurement:

- Does the `rubband` Windows wheel expose `LiveShifter`, and what is its start delay and CPU load at 48 kHz
  with 256/512-frame blocks? Fallback: `python-stretch`.
- Measured glass-to-glass latency mic → app → `CABLE Input` → `CABLE Output` on Michael's machine.
- Does Pedalboard's block-wise `process(reset=False)` behave for reverb/EQ in a live callback (the
  streaming bugs reported are in `PitchShift`, which we do not use live)?
- PyInstaller bundle size and startup time for the DSP-only app on Windows with Pedalboard + `rubband`.
- Michael's GPU (only matters when the Neural Engine epic is scheduled).

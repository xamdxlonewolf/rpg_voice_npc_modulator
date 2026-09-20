# Neural voice — what is real, what is not (E7)

Status: 2026-09-20 (updated after the GPU slice). Constraints unchanged: free, fully local, no paid APIs, no cloud
(ADR-0001). This page is the honest answer to "make me sound like someone else" and
"give me an Irish accent", and what the app ships for it today.

## The short version

| You want | Free + local today? | Hardware | In this build |
| --- | --- | --- | --- |
| Shape *how you sound* (deeper, tiny, gravelly, ghostly, hollow, distant, breathy) | **Yes** | Any CPU | DSP Engine sliders, Tone Tags, **Presets**, **Design from Tone Hints** |
| Describe a voice in words and get usable sliders | **Yes** (rules, not a model) | Any CPU | **Design from Tone Hints** — offline lexicon → tags + sliders |
| Sound like a genuinely *different person* | Only with a neural voice converter | **NVIDIA GPU, ≈ 6 GB VRAM**, 300–500 ms latency; CPU ≈ 1–2 s (unusable live) | **`NeuralEngine` (X-VC) in Preview** once the opt-in pack is downloaded; Roleplay not yet wired; unverified on hardware |
| Describe a person in words and get *that person* | Only with a voice-design TTS feeding a converter | NVIDIA GPU, multi-GB download | **Neural Voice design (Qwen3-TTS VoiceDesign)** when both packs are installed; lexicon fallback otherwise, and it says so |
| Speak with an **Irish / British accent** in *your* live voice | **No** — see below | — | **Not faked.** Accent words are kept as notes only |
| A designed character *reading text* in an accent (TTS, not your voice) | Plausible with a local TTS that takes accent prompts | GPU for good ones; small CPU TTS exists | Not shipped; would be a Preview/soundboard feature, not Roleplay |

## Why live accent conversion is not on the table

Voice conversion (X-VC, Seed-VC, RVC, LLVC) changes **timbre**: the model re-renders the
frames of your speech with another speaker's voice quality while keeping *your* phonemes,
timing and intonation. An accent lives in exactly the parts it keeps: which vowels you
use, where the stress falls, how the sentence melody moves, which consonants you drop or
tap. Converting to an Irish speaker's reference clip gives you *an Irish person's voice
quality saying the words the way you said them* — a Dubliner with a Birmingham accent, not
a Dubliner. "Accent conversion" is a separate research area (phoneme- and prosody-level
re-synthesis); the open models for it are not real-time, not robust on arbitrary speech,
and none of the free local ones are usable for live tabletop play. Anything we shipped
under that label would be pretending.

What does work for accents: the GM does the accent (acting), and the DSP or a future
Neural Engine changes the *voice* around it. Prompted TTS models can *speak* in an accent
from a text prompt, but that produces a synthetic narrator reading typed lines — it is not
your live voice and it is not Roleplay Mode.

## What runs where

| Piece | CPU-only laptop | NVIDIA GPU |
| --- | --- | --- |
| DSP Engine (pitch, Body formant shift, Tonality, Growl, Hollow, Room, Distance, Breath, Gate) | Yes, live and Preview | Same |
| Presets and Design from Tone Hints (lexicon) | Yes, instant | Same |
| Zero-shot voice conversion (X-VC, MIT; Seed-VC, GPL-3.0) | No for live: ~1–2 s | Yes: ~3–6 GB VRAM, 300–500 ms |
| Voice design from text (Qwen3-TTS VoiceDesign, Apache-2.0) | Too slow to be pleasant; multi-GB | Yes; reference clip in seconds |
| Small CPU TTS (Kokoro, Chatterbox-Nano, MOSS-TTS-Nano) | Yes, for a *synthetic* Preview voice only | Same |

Runtime cost of the neural path if it is ever installed: CUDA PyTorch ≈ 2–2.5 GB plus
model weights; call it 5–8 GB on disk, downloaded once into the user-data folder, NVIDIA
only, opt-in with a visible size before anything starts. Nothing in this build downloads
anything.

## What shipped in this epic

### CPU slice (PR #16, #17)

- **Presets** (`votr.presets`): twelve named starting points; "Use preset" opens your
  saved Voice for that preset or a fresh copy of the bundled recipe.
- **Design from Tone Hints** (`votr.voicedesign.LexiconDesigner`): free text → Tone
  Tags and sliders offline, instantly, on any CPU. Accent words and "sound like X" get a
  note instead of a fake.

### GPU slice (this PR)

- **`NeuralEngine`** (`votr.neural_engine`, `engine_id = "neural-v0"`): a real Engine
  behind the ADR-0004 seam. It owns 48 kHz ↔ 16 kHz resampling, X-VC's streaming
  geometry (2.4 s window = history | 240 ms current | 20 ms smooth | 100 ms future),
  the crossfade, a dry/wet `mix` with a latency-matched dry path, and a `render` that
  pushes the Take through the same `process_block` path so Preview equals live
  (ADR-0005). `latency_frames()` reports ≈ 360 ms + resampler delay. Capabilities:
  `changes_identity=True`, `requires_gpu=True`. The model sits behind a five-line
  `VoiceConverter` protocol; tests drive it with a fake.
- **Model adapters** (`votr.neural_backends`): `XvcConverter` calls X-VC's own
  `bins/infer_utils` (`load_xvc`, `precompute_conditions`, `run_stream_chunk_forward`)
  from the downloaded source checkout, rewriting its yaml to point at the downloaded
  tokenizer and speaker encoder. `QwenVoiceDesign` calls
  `qwen_tts.Qwen3TTSModel.generate_voice_design`. **Neither has run on real hardware
  in this project yet** — see blockers.
- **Opt-in model packs** (`votr.neural_pack`, Settings → Neural): two packs with every
  file's publisher URL, size, licence and (where published) SHA-256. Download resumes
  from `.part` files, verifies checksums, unzips the source archive, can be cancelled,
  and never starts until the GM ticks the licence acknowledgement. Buttons are disabled
  without an NVIDIA GPU of ≥ 6 GiB. Nothing is bundled or redistributed.
- **Gate** (`votr.neural`): `probe_runtime` = GPU (nvidia-smi) + CUDA PyTorch importable
  + pack installed. `Session.engine_installed("neural-v0")` is true only when all three
  hold; Neural Voices in the Library stay disabled otherwise; `preview_engine()` routes a
  Neural draft to the Neural Engine and falls back to DSP if it fails to start.
- **Neural Voice design**: when both packs and the runtime are present, "Design from
  Tone Hints" speaks a reference clip with Qwen3-TTS VoiceDesign, saves it under
  `neural/clips/`, and the draft becomes a Neural Voice pointing at it. If the neural
  designer is off *or blows up*, the lexicon designer answers and says why.

## Model packs and licences (read before downloading)

| Pack | Files | Size | Licence |
| --- | --- | --- | --- |
| X-VC conversion | X-VC source at commit `49df8c5` (GitHub zip) | small | MIT |
| | `xvc.pt` (chenxie95/X-VC) | 5.0 GB | MIT |
| | GLM-4-Voice tokenizer (`zai-org/glm-4-voice-tokenizer`) | 1.46 GB | **custom glm-4-voice licence** — free for personal and research use; commercial use requires registration with Zhipu AI; products must show "Built with glm-4"; governed by PRC law. Not OSI. X-VC does not run without it. |
| | ERes2Net speaker encoder (ModelScope `iic/speech_eres2net_sv_en_voxceleb_16k`) | 27 MB | Apache-2.0 |
| Qwen3-TTS VoiceDesign | `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` | 4.5 GB | Apache-2.0 |

For a GM using the app at their own table the GLM-4-Voice terms are met by the licence
copy the pack downloads and the notice shown in Settings → Neural. Anyone selling a
service on top of it would have to register with Zhipu. If that is unacceptable, the
alternative zero-shot converter is Seed-VC (GPL-3.0, archived), which would need its
own adapter.

## What a GM can do

| | No NVIDIA GPU (most laptops) | NVIDIA GPU ≥ 6 GiB, packs installed |
| --- | --- | --- |
| Sliders, Tone Tags, Presets, Preview, Roleplay | Yes | Yes |
| Design from Tone Hints | Lexicon → tags + sliders | Qwen3-TTS reference clip → Neural Voice (falls back to lexicon if the model fails) |
| Sound like a different person | No | Neural Engine, ≈ 360 ms + device latency in Preview |
| Neural Voice in Roleplay Mode | No | **Not wired yet** — Roleplay still runs the DSP Engine (out of scope for this slice) |
| Live Irish/British accent on your own speech | No | No — and we will not pretend |
| Settings → Neural | Honest status, buttons disabled | Size + licences + download/cancel/resume |

## Installing the runtime (learned on Michael's Windows box, Python 3.13)

Run from source, not the installer. In the repo folder:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -e ".[neural]"
python -c "import torch, transformers; print(torch.__version__, torch.cuda.is_available(), transformers.__version__)"
```

The last line must print a `+cu1xx` torch, `True`, and transformers ≥ 4.46. Use `cu126`
if `nvidia-smi` reports CUDA 12.6. Then `python -m votr` → Settings → Neural → tick the
licence acknowledgement → Download.

What we learned the hard way:

- X-VC pins `torch==2.5.1` and `transformers==4.44.1`. Neither installs on Python 3.13
  (no torch wheels; `transformers` 4.44 drags an old `tokenizers` that tries to compile
  Rust and PyO3 refuses). torch 2.9 + transformers ≥ 4.46 install fine; whether X-VC's
  inference is happy on them is the next thing Michael's box will tell us.
- X-VC's `utils/log.py` imports `wandb`, `tensorboard` and `matplotlib` at load time and
  `model.py` imports `audiotools` (descript-audiotools); missing any of them surfaces as
  hydra's opaque *"Error locating target 'models.codec.sac.model.XVC'"*. The `neural`
  extra now includes them, and `XvcConverter` pre-imports the model module so the real
  error and a `pip install` hint appear in the editor and the log.
- `SoX could not be found` is a warning from descript-audiotools' optional SoX path; it
  is not needed. `flash-attn is not installed` is Qwen3-TTS choosing the slower
  attention; also fine.
- Qwen3-TTS VoiceDesign generated a clip on the first try (torch 2.9, transformers ≥
  4.46, `qwen-tts` 0.1.1).

## Remaining blockers for Michael's box

1. **First successful X-VC load.** Runtime is now installable; the next report will show
   whether `models.codec.sac.model` imports and `load_xvc` runs on torch 2.9 /
   transformers ≥ 4.46 with the rewritten yaml. Fixes belong in
   `votr/neural_backends.py`.
2. **Roleplay Mode** still uses the DSP Engine for Neural Voices (S7.4 next slice).
3. **No reference-clip picker** in the editor; a Neural Voice is made by "Design from
   Tone Hints" (with the VoiceDesign pack) or by editing the Voice JSON.
4. **Installer** does not carry the runtime; source checkout required.
5. **Quality and latency numbers** are the publishers'; nothing measured here yet.

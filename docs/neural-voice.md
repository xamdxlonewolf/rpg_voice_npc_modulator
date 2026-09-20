# Neural voice — what is real, what is not (E7)

Status: 2026-09-20. Constraints unchanged: free, fully local, no paid APIs, no cloud
(ADR-0001). This page is the honest answer to "make me sound like someone else" and
"give me an Irish accent", and what the app ships for it today.

## The short version

| You want | Free + local today? | Hardware | In this build |
| --- | --- | --- | --- |
| Shape *how you sound* (deeper, tiny, gravelly, ghostly, hollow, distant, breathy) | **Yes** | Any CPU | DSP Engine sliders, Tone Tags, **Presets**, **Design from Tone Hints** |
| Describe a voice in words and get usable sliders | **Yes** (rules, not a model) | Any CPU | **Design from Tone Hints** — offline lexicon → tags + sliders |
| Sound like a genuinely *different person* live | Only with a neural voice converter | **NVIDIA GPU, ≈ 6 GB VRAM**, 300–500 ms latency; CPU ≈ 1–2 s (unusable live) | **Not shipped.** Seam + GPU detection only (Settings → Neural) |
| Describe a person in words and get *that person* | Only with a voice-design TTS feeding a converter | NVIDIA GPU, multi-GB download | **Not shipped.** Design falls back to the lexicon and says so |
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

- **Presets** (`src/votr/assets/presets.json`, `votr.presets`): twelve named starting
  points (Orc Warchief, Ancient Crone, Cave Troll, Pixie, Restless Ghost, Clockwork
  Automaton, Court Herald, Shadow Whisperer, Elder Dragon, Kobold Scout, Lich, Through a
  Helmet). Each is Tone Tags plus a slider recipe. "Use preset" in the editor loads a new
  unsaved draft you Preview, tune and Save. Presets are copies, never links.
- **Design from Tone Hints** (`votr.voicedesign.LexiconDesigner`): reads the free-text
  Tone Hints and sets Tone Tags and sliders offline, instantly, on any CPU. Words pick
  Macros (`gravelly`, `giant`, `ghostly`, …), other words nudge sliders (`deep`, `bright`,
  `echoing`, `distant`, `breathy`, `angry`), intensifiers scale them (`very`, `slightly`),
  and `-5 semitones` is taken literally. It reports what it heard. Accent words and
  "sound like X" are recognised and answered with a note instead of a fake.
- **Neural seam** (`votr.neural`): `detect_nvidia_gpu()` via `nvidia-smi`, a plain-language
  status in **Settings → Neural**, `NEURAL_ENGINE_ID` reserved and *not* in the installed
  Engine list (Voices for it stay disabled), and `NeuralDesigner` first in the designer
  chain reporting exactly why it is unavailable. `design_voice()` falls back to the
  lexicon. No download button exists yet.

## What would come next (only with a GPU on Michael's desk)

S7.1 opt-in Engine pack download with checksum and resume → S7.3 Qwen3-TTS VoiceDesign
reference clip from Tone Hints → S7.2 X-VC zero-shot `Engine` (`changes_identity=True`,
`requires_gpu=True`) → S7.4 Library badges and latency warning. Details and licences in
`docs/backlog.md` E7 and `docs/discovery-local-voice.md` §B.

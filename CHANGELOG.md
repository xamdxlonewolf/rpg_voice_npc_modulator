# Changelog

All notable changes to Voice of the Realm are listed here.
Windows installer artifacts are attached to GitHub Releases on `v*` tags.

## Unreleased

- Mimic clip library for Neural Voices: Engine dropdown in the editor (DSP ↔
  Neural), record up to 30 s with a countdown, upload WAV/FLAC/MP3/OGG/AIFF,
  pick/rename/delete clips, Voices store the clip id. No more hand-editing JSON.

- Neural: `pip install -e ".[neural]"` extra with the runtime X-VC and Qwen3-TTS
  actually need (incl. wandb/tensorboard/matplotlib/audiotools that X-VC imports
  at load); the adapter now reports the real import error and a pip hint instead
  of hydra's "Error locating target"; the editor shows why the Engine failed.

- E7 GPU slice: a real `NeuralEngine` (X-VC zero-shot voice conversion) behind the
  Engine seam with streaming windows, latency-matched mix and Preview == live;
  opt-in model pack download in Settings → Neural (sizes, licences incl. the
  GLM-4-Voice caveat, resume, checksum, cancel; NVIDIA ≥ 6 GiB only); neural Voice
  design from Tone Hints via Qwen3-TTS VoiceDesign when installed, lexicon
  fallback otherwise. Nothing bundled; unverified on real hardware so far.

- Presets: a Voice saved from a preset now remembers it (`preset` field), so
  choosing that preset again opens your saved Voice instead of a new bundled
  copy; "Fresh copy" starts over from the bundled recipe under a new name.
  Orc Warchief, Cave Troll, Elder Dragon, Lich, Through a Helmet and Clockwork
  Automaton toned down.

- E7 first slice: twelve bundled Voice **Presets** ("Use preset" in the editor),
  **Design from Tone Hints** (offline word lexicon → Tone Tags + sliders, reports
  what it heard, notes accent / "sound like" requests instead of faking them),
  NVIDIA GPU detection with an honest **Settings → Neural** page, and the neural
  designer/Engine seam. Nothing neural is installed or downloaded. See
  `docs/neural-voice.md`.

- Breath is no longer a noise layer. It is a whispered copy of the voice itself
  (random-phase STFT of the Take, voice-shaped, frame-tight), so it only exists
  where the voice does; silence and a room floor gain nothing.

- Preview sliders retuned from the Windows listen: Body is a real formant shift,
  Tonality a level-matched tilt, Growl level-matched saturation with rasp, Hollow a
  cupped mid-band cavity (0 = off), Distance a gentler darker-and-quieter curve,
  Breath aspiration that follows the voice (no hiss bed), Gate an RMS-envelope
  gate with hold and hysteresis that never chops speech. The effect chain is
  level-matched to the dry Take on `render`; Room is unchanged.

## 0.1.0 — 2026-09-20

Rough-draft app (E0–E6):

- Create, save, and edit Voices (Tone Hints, Tone Tags, DSP Engine sliders).
- Voice Library and Active Voice buttons.
- Delayed Preview: record a Take, render through the Engine, play speakers.
- Roleplay Mode into a Virtual Cable; Discord setup wizard (link VB-CABLE, never bundle).
- First-run setup, Latency Test, and Settings.
- PyInstaller `--onedir` spec and per-user Inno Setup script.
- GitHub Actions release pipeline on version tags.

Glass-to-glass latency, Windows bundle size, and cold-start are **not** measured
in this changelog. See `docs/spikes/latency.md` and `docs/spikes/pyinstaller.md`.

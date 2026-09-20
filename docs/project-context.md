# Voice of the Realm — Project Context

Owner: Michael Cobb. Repo: [xamdxlonewolf/rpg_voice_npc_modulator](https://github.com/xamdxlonewolf/rpg_voice_npc_modulator) (currently empty apart from README/LICENSE).

## Goal

A free, installable, fully local desktop app for a tabletop Game Master who runs games
online. The GM creates character Voices, saves each one as a clickable button, previews
it, and then speaks through the selected Voice into Discord (or any online tool). In
Roleplay Mode the GM's own dry voice is never sent to the output; only the processed
Character Voice reaches the Virtual Cable that Discord uses as its microphone.

Rough-draft target (delivered by the backlog in `docs/backlog.md`):

1. Create a Voice (name, Tone Hints, Tone Tags → slider Macros, sound parameters).
2. Save it; it appears as a Voice Button in the Voice Library.
3. Click it, record a Take, hear it played back through the Voice on your speakers
   (delayed Preview), tune, repeat.
4. Enter Roleplay Mode: mic → DSP Engine → Virtual Cable → Discord, live.

## Hard constraints

| Constraint | Meaning for the design |
| --- | --- |
| Free for users, free to develop | No paid APIs, no subscriptions, no commercial SDK licences. Only free/open-source libraries and openly licensed models. |
| Fully local | No cloud storage, no cloud inference. Voices live on the user's disk. Works offline after install. |
| Installable desktop app | A real installer, not "clone the repo and pip install". Windows 10/11 first. |
| Preview before use | The GM must be able to hear a Voice before selecting it for play (record-then-play). |
| Roleplay Mode | The dry signal must never be written to the output; only the Character Voice is. |
| Python | Python + PySide6 + JSON + PyInstaller. Confirmed. |

## Locked decisions

Decisions that met all three ADR gates (hard to reverse, surprising without context,
real trade-off) are ADRs. The rest are listed after.

| ADR | Decision |
| --- | --- |
| `docs/adr/0001-local-only-free-no-cloud.md` | Local-only and free; no cloud or paid AI services. |
| `docs/adr/0002-roleplay-mode-online-only-via-virtual-cable.md` | Roleplay Mode is online-only, delivered through a Virtual Cable. In-person table play is not a target. |
| `docs/adr/0003-app-licensed-under-gpl.md` | The app is GPL, so Rubber Band / Pedalboard may be used in the live path. |
| `docs/adr/0004-dsp-first-engine-behind-engine-interface.md` | DSP modulation of the GM's own voice now; a Neural Engine ("different person", GPU) later behind the same Engine interface. Tone Hints = notes + Tone Tags → Macros; free text kept for later. |
| `docs/adr/0005-delayed-preview-record-then-play.md` | Preview records a Take and plays it back through the Voice on the GM's speakers; not a live monitor. Preview renders through the same streaming Engine path as Roleplay Mode. |
| `docs/adr/0006-link-to-not-bundle-virtual-cable-drivers.md` | Virtual Cable drivers are detected and linked to via a wizard, never bundled or silently installed. |

Other settled points (grill round 1, `docs/grill-round-1.md`):

- Target machine: Windows 10/11 first, CPU-only baseline, NVIDIA GPU optional and only
  relevant to the later Neural Engine epic. Code stays cross-platform; installer is
  Windows-only for the rough draft.
- Latency: auto-tune on first run, default ~120 ms, manual override. Online-only play
  makes this a quality target rather than a hard ceiling.
- Stack: Python 3.11/3.12, `sounddevice`, Rubber Band `LiveShifter` (via `rubband`) for
  live pitch/formant with Signalsmith Stretch (`python-stretch`) as fallback, NumPy/SciPy
  and Pedalboard for effects, PySide6 UI, one JSON file per Voice in the user-data folder,
  PyInstaller `--onedir` + Inno Setup. Details in `docs/discovery-local-voice.md`.

## Assumed defaults (routine calls, override by editing this list)

- Preview Take is push-to-talk: hold to record, release to render and play. A bundled
  sample phrase is used when no mic is present.
- Roleplay Mode has an optional monitor toggle (Character Voice to the GM's headphones),
  off by default.
- Voice import/export = copy the JSON file; no UI for it in the rough draft.
- Discord wizard targets VB-CABLE first; other cables detected by device-name pattern.
- Panic/mute hotkey is global and always available in Roleplay Mode.
- Voice creation helpers are CPU-only and offline: bundled **Presets** (named slider
  recipes + Tone Tags) and **Design from Tone Hints** (free text → Tone Tags + sliders by
  a word lexicon). They shape how the GM sounds; they cannot produce a specific person or
  an accent. A neural designer sits first in the chain and falls back when not installed.

## Explicitly out of scope for the rough draft

- In-person table play, room speakers, feedback suppression (ADR-0002).
- Bundling anything neural. The Neural Engine (`neural-v0`, X-VC), the opt-in model
  pack download and the neural Voice designer **are** in (E7); they run only on an
  NVIDIA GPU ≥ 6 GiB with CUDA PyTorch and the packs the GM chose to download. Roleplay
  Mode with a Neural Voice is not wired yet (`docs/neural-voice.md`).
- Live accent conversion of the GM's own speech. Voice conversion moves timbre, not
  pronunciation; the app will not pretend otherwise (`docs/neural-voice.md`).
- Cloud sync, accounts, sharing marketplace.
- Training custom neural models inside the app.
- Music/soundboard features, ambience, session recording.
- macOS/Linux installers; mobile.

## Where things are

| Doc | Purpose |
| --- | --- |
| `docs/project-context.md` | This file: goals, constraints, locked decisions, defaults. |
| `docs/discovery-local-voice.md` | Technical findings: approaches, stack, feasibility, risks. |
| `docs/grill-round-1.md` | Phase-1 grill, answered; overrides noted. |
| `docs/CONTEXT.md` | Glossary (domain language only). |
| `docs/adr/` | Architecture decision records 0001–0006. |
| `docs/backlog.md` | Rough-draft backlog: epics → stories → tasks, plus the later Neural Engine epic. |
| `docs/neural-voice.md` | E7: what is feasible free and local (CPU vs NVIDIA GPU), why live accent conversion is not, what shipped. |

## Implementation tooling

Implementation work in this repository uses Grok 4.6 high fast.

# Voice of the Realm

A free, local desktop app for a tabletop Game Master who plays online: create
character Voices, preview them, and speak through the selected Voice into Discord.

Product decisions live in [`docs/`](docs/project-context.md). The app is GPL-3.0;
see [`LICENSE`](LICENSE) and [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
Windows packaging lives in `packaging/VoiceOfTheRealm.spec` and
`installer/votr.iss`. The installer is built on a `v*` tag (GitHub Actions,
`windows-latest`) and ships the **DSP** app only (Voices, Preview, Roleplay).
CUDA / X-VC are not in that exe — see [`docs/install-windows.md`](docs/install-windows.md).
Virtual Cable drivers are never bundled.

## Requirements

- Python 3.11 or newer

## Install and run locally

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
python -m votr
```

That opens the **Voice of the Realm** window. In the editor, "Use preset" loads a
named starting point and "Design from Tone Hints" turns a description like *a
gravelly old dwarf in a great hall* into Tone Tags and sliders — offline, CPU-only.

**Windows users who only need DSP** should prefer the GitHub Release installer
over cloning this repo. Neural still needs the venv extra above plus
`pip install -e ".[neural]"` — [`docs/install-windows.md`](docs/install-windows.md)
and [`docs/neural-voice.md`](docs/neural-voice.md).

## Tests and lint

```bash
pytest
ruff check src tests
```

Tests construct the window on Qt's offscreen platform so they can run without a
display. `python -m votr --headless` (or `VOTR_HEADLESS=1`) selects that plugin,
builds the window, and exits without entering the event loop. GUI tests skip if
PySide6 cannot load its native libraries.

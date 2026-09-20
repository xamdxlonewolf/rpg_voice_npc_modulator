# Voice of the Realm

A free, local desktop app for a tabletop Game Master who plays online: create
character Voices, preview them, and speak through the selected Voice into Discord.

This repository is at the S0.1 scaffold — packaging, tests, and an empty window.
Product decisions live in [`docs/`](docs/project-context.md). The app is GPL-3.0;
see [`LICENSE`](LICENSE) and [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## Requirements

- Python 3.11 or newer

## Install and run locally

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
python -m votr
```

That opens an empty PySide6 window titled **Voice of the Realm**.

## Tests and lint

```bash
pytest
ruff check src tests
```

Tests construct the window on Qt's offscreen platform so they can run without a
display. `python -m votr --headless` (or `VOTR_HEADLESS=1`) builds the window and
exits without entering the event loop.

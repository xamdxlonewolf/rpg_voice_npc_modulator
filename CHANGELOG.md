# Changelog

All notable changes to Voice of the Realm are listed here.
Windows installer artifacts are attached to GitHub Releases on `v*` tags.

## Unreleased

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

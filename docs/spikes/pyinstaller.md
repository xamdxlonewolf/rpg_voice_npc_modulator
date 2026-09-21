# Spike S0.6 — PyInstaller bundle

Date: 2026-09-20. **Windows size and cold-start are not measured.**

## Planned check

Build `--onedir` collecting `rubband` (optional), `pedalboard`, PySide6 plugins
(not Addons), `python-stretch`, and `sounddevice` (PortAudio DLL). Record
installer-folder size and cold-start to the empty window on Windows 10/11.

Helpers:

- `packaging/VoiceOfTheRealm.spec` (DSP onedir; torch / X-VC excluded)
- `python scripts/build_bundle.py`
- `python scripts/smoke_headless.py` (and `--frozen` after a build)
- `installer/votr.iss` (per-user Inno Setup; version from `pyproject.toml`)
- `installer/what-this-installs.txt` (shown before files copy; DSP vs Neural)
- `docs/install-windows.md` (what the installer contains; Neural extra)

## Linux cloud VM (honest, not a Windows result)

An earlier `PyInstaller --collect-all PySide6` smoke on this VM produced
`dist/VoiceOfTheRealm` at **722 MB**. That figure is **Linux-only** and is
why the E6 spec does **not** use `--collect-all PySide6`.

Do **not** copy 722 MB or the old 0.19 s `--headless` time into a Windows
estimate. Windows bundle size and cold-start have not been measured.

- `rubband` has **no published Windows wheel**; the first Windows installer
  should ship `python-stretch` and collect Rubber Band only if a wheel or
  vendored binary is present (see `docs/spikes/pitch-core.md`).

## Blocked on Windows

- Bundle size with `rubband` + `pedalboard` + PySide6 on Windows 10/11.
- Cold-start time to "Voice of the Realm" after a user double-click.
- Whether a `rubband` native DLL is present after collect.
- Inno Setup compile (`iscc`) — not installed on this Linux VM.
- S6.4 clean-machine acceptance (see `docs/spikes/acceptance-rough-draft.md`).

Re-run `python scripts/build_bundle.py` then `iscc installer/votr.iss` on a
Windows 10/11 machine. GitHub Actions on a `v*` tag uses `windows-latest` for
that path.

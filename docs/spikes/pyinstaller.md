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

Re-run 2026-09-21 on this VM (`python scripts/build_bundle.py` then
`python scripts/smoke_headless.py --frozen`):

- `dist/VoiceOfTheRealm` **239 MB** onedir (PySide6 hooks only — not
  `--collect-all`, and not `collect_data_files("PySide6")` of the whole
  package). An earlier collect-all smoke on this VM was 722 MB; a mistaken
  full-package collect in this branch first landed at 756 MB before the
  spec was tightened.
- Frozen `--headless` **exit 0**. No `torch` / X-VC / qwen in the tree.
- `sounddevice` warned that PortAudio was missing here; that is Linux-VM
  only. Windows GHA / Michael's box should pick up the PortAudio DLL.
- `LICENSE`, notices, and `what-this-installs.txt` land in `_internal/`;
  Inno also copies them next to the exe.

Do **not** copy 239 MB, 722 MB, or 756 MB into a Windows estimate.
Windows bundle size and cold-start have not been measured.

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

# Spike S0.6 — PyInstaller bundle

Date: 2026-09-20. **Windows size and cold-start are not measured.**

## Planned check

Build `--onedir` collecting `rubband`, `pedalboard`, PySide6, `python-stretch`,
and `sounddevice` (PortAudio DLL). Record installer-folder size and cold-start
to the empty window on Windows 10/11.

Helper: `python scripts/build_bundle.py`.

## Linux cloud VM (honest, not a Windows result)

`python scripts/build_bundle.py` succeeded here (`PyInstaller --onedir
--collect-all PySide6`).

| Fact | Linux VM only |
| --- | --- |
| `dist/VoiceOfTheRealm` size | **722 MB** (`--collect-all PySide6` pulls Addons; too fat to ship) |
| `VoiceOfTheRealm --headless` | exit 0 in **0.19 s** (offscreen Qt, not a user double-click) |

Do **not** copy 722 MB or 0.19 s into a Windows estimate. The next Windows
build should collect `PySide6` essentials only (not Addons).

- `rubband` has **no Windows wheel**; a Windows bundle cannot collect it via
  `pip` until we vendor a wheel (see `docs/spikes/pitch-core.md`).

## Blocked on Windows

- Bundle size with `rubband` + `pedalboard` + PySide6.
- Cold-start time to "Voice of the Realm".
- Whether the `rubband` native DLL is present after `collect_all`.

Re-run `scripts/build_bundle.py` on a Windows 10/11 machine with the S1.2
deps installed. If `rubband` still has no wheel, the first installer should
ship `python-stretch` and treat Rubber Band as an optional collected binary.

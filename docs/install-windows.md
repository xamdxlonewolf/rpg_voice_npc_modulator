# Windows install (DSP app)

This is the shippable install path for the **DSP** Game Master loop: Voices,
Preview, Roleplay into a Virtual Cable. You do not need `git clone` for that
loop once a `v*` GitHub Release has an installer attached.

## What the installer contains

The Inno Setup per-user installer (`VoiceOfTheRealm-<version>-setup.exe`)
copies the PyInstaller `--onedir` folder:

- `VoiceOfTheRealm.exe` plus the DSP runtime (PySide6, NumPy, Pedalboard,
  `python-stretch`, PortAudio via sounddevice, bundled `macros.json` /
  `presets.json` / `sample_take.wav`)
- `LICENSE` (GPL-3.0-or-later) and `THIRD_PARTY_NOTICES.md`
- `what-this-installs.txt` (the same honesty page the wizard shows first)

It also creates a **Start Menu** shortcut and an **uninstaller**. Version is
stamped from `pyproject.toml`. Virtual Cable drivers are **not** bundled
(ADR-0006); first-run / Discord setup links to VB-CABLE.

It does **not** contain CUDA, PyTorch, X-VC, or the multi-GB model packs.
Those cannot honestly live in this frozen folder.

## Install (Michael's box)

1. On Windows 10/11, download the setup exe from the GitHub Release for a
   `v*` tag (Actions builds it on `windows-latest`).
2. Run it (per-user, no admin). Read the GPL page and the “what this
   installer includes” page.
3. Launch from the Start Menu. First-run setup: devices, Latency Test,
   Discord wizard.
4. Create a Voice, Preview a Take, enter Roleplay Mode.

This Linux cloud VM cannot compile Inno Setup or run the clean-machine
checklist. Those remaining steps are in
[`docs/spikes/pyinstaller.md`](spikes/pyinstaller.md) and
[`docs/spikes/acceptance-rough-draft.md`](spikes/acceptance-rough-draft.md).

## Neural Engine (not in the installer)

Neural still needs a **source checkout** and a venv:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -e ".[neural]"
python -m votr
```

Then Settings → Neural → licences → Download. Details:
[`docs/neural-voice.md`](neural-voice.md).

The Start Menu DSP app will not load X-VC. When Neural *is* enabled in the
source venv (GPU + pack, or a saved Neural Voice), launch shows a
**Loading Neural Engine…** overlay and builds X-VC off the UI thread so
Windows does not paint Not Responding. DSP-only launches skip that load.

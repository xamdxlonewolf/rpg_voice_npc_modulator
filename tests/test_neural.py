# Copyright (C) 2026 Michael Cobb
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest

from votr import neural
from votr.engine import Engine
from votr.neural import (
    MIN_VRAM_MIB,
    GpuInfo,
    NeuralDesigner,
    create_neural_engine,
    detect_nvidia_gpu,
    neural_status,
    parse_nvidia_smi,
    probe_runtime,
)
from votr.neural_engine import NEURAL_ENGINE_ID, REFERENCE_CLIP_KEY, NeuralEngine
from votr.neural_pack import CONVERSION_PACK, VOICE_DESIGN_PACK, pack_root
from votr.session import INSTALLED_ENGINE_IDS, Session
from votr.spikes.pitch_core import stretch_available
from votr.voice import DSP_ENGINE_ID
from votr.voicedesign import design_voice
from votr.wavutil import write_wav

BIG_GPU = GpuInfo("NVIDIA GeForce RTX 3060", 12288)


def _installed_pack(root: Path, pack) -> None:
    """Fake a downloaded pack: every file present at its declared size."""
    for item in pack.files:
        if item.unzip_to:
            (root / item.unzip_to).mkdir(parents=True, exist_ok=True)
            (root / item.unzip_to / "README.md").write_text("x")
            continue
        path = root / item.relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            handle.truncate(item.size or 0)


class FakeConverter:
    sample_rate = 16000

    def __init__(self) -> None:
        self.reference: tuple[int, int] | None = None
        self.calls = 0

    def set_reference(self, clip: np.ndarray, sample_rate: int) -> None:
        self.reference = (clip.size, sample_rate)

    def convert_window(self, window: np.ndarray) -> np.ndarray:
        self.calls += 1
        return -window


# --- GPU detection -----------------------------------------------------------


def test_parse_nvidia_smi_csv() -> None:
    gpu = parse_nvidia_smi("NVIDIA GeForce RTX 3060, 12288 MiB\n")
    assert gpu == GpuInfo(name="NVIDIA GeForce RTX 3060", vram_mib=12288)
    assert gpu.enough_vram
    assert parse_nvidia_smi("") is None
    assert parse_nvidia_smi("garbage line") is None
    small = parse_nvidia_smi("NVIDIA GeForce GTX 1050, 2048 MiB")
    assert small is not None and not small.enough_vram


def test_detect_without_nvidia_smi_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(neural.shutil, "which", lambda _name: None)
    assert detect_nvidia_gpu() is None


def test_detect_parses_a_fake_driver(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(neural.shutil, "which", lambda _name: "/fake/nvidia-smi")

    def fake_run(*_args, **_kwargs):
        return subprocess.CompletedProcess(
            args=[], returncode=0, stdout="NVIDIA RTX A4000, 16376 MiB\n", stderr=""
        )

    monkeypatch.setattr(neural.subprocess, "run", fake_run)
    assert detect_nvidia_gpu() == GpuInfo("NVIDIA RTX A4000", 16376)


def test_detect_tolerates_a_broken_driver(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(neural.shutil, "which", lambda _name: "/fake/nvidia-smi")

    def boom(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=1)

    monkeypatch.setattr(neural.subprocess, "run", boom)
    assert detect_nvidia_gpu() is None


# --- runtime gate ------------------------------------------------------------


def test_probe_can_skip_torch_when_a_gpu_is_present(tmp_path: Path) -> None:
    calls = []

    def torch_state():
        calls.append(1)
        return True, True, "should not be asked"

    runtime = probe_runtime(
        tmp_path,
        gpu=BIG_GPU,
        torch_state=torch_state,
        check_torch=False,
    )
    assert runtime.gpu_ok
    assert not runtime.ready
    assert calls == []
    assert "deferred" in runtime.torch_detail


def test_probe_on_a_machine_without_gpu_is_off_and_skips_torch(tmp_path: Path) -> None:
    calls = []

    def torch_state():
        calls.append(1)
        return True, True, "should not be asked"

    runtime = probe_runtime(tmp_path, detect=lambda: None, torch_state=torch_state)
    assert runtime.gpu is None
    assert not runtime.ready and not runtime.design_ready
    assert runtime.reason == "no NVIDIA GPU detected"
    assert calls == []
    assert runtime.root == pack_root(tmp_path)


def test_probe_reasons_in_order(tmp_path: Path) -> None:
    small = probe_runtime(tmp_path, gpu=GpuInfo("GTX 1050", 2048))
    assert not small.ready and str(MIN_VRAM_MIB) in small.reason
    no_torch = probe_runtime(
        tmp_path, gpu=BIG_GPU, torch_state=lambda: (False, False, "PyTorch missing")
    )
    assert not no_torch.ready and no_torch.reason == "PyTorch missing"
    no_pack = probe_runtime(
        tmp_path, gpu=BIG_GPU, torch_state=lambda: (True, True, "PyTorch 2.5 with CUDA")
    )
    assert not no_pack.ready and "not downloaded" in no_pack.reason
    _installed_pack(pack_root(tmp_path), CONVERSION_PACK)
    ready = probe_runtime(
        tmp_path, gpu=BIG_GPU, torch_state=lambda: (True, True, "PyTorch 2.5 with CUDA")
    )
    assert ready.ready and ready.reason == ""
    assert not ready.design_ready
    _installed_pack(pack_root(tmp_path), VOICE_DESIGN_PACK)
    assert probe_runtime(
        tmp_path, gpu=BIG_GPU, torch_state=lambda: (True, True, "ok")
    ).design_ready


def test_create_engine_returns_none_when_not_ready_or_backend_fails(
    tmp_path: Path,
) -> None:
    off = probe_runtime(tmp_path, detect=lambda: None)
    assert create_neural_engine(off) is None
    _installed_pack(pack_root(tmp_path), CONVERSION_PACK)
    ready = probe_runtime(tmp_path, gpu=BIG_GPU, torch_state=lambda: (True, True, "ok"))

    def broken(_root):
        raise RuntimeError("no such module")

    assert create_neural_engine(ready, converter_factory=broken) is None
    engine = create_neural_engine(
        ready, converter_factory=lambda _root: FakeConverter()
    )
    assert isinstance(engine, NeuralEngine)
    assert isinstance(engine, Engine)
    assert engine.engine_id == NEURAL_ENGINE_ID
    assert engine.capabilities().changes_identity and engine.capabilities().requires_gpu


def test_status_is_honest_about_hardware_and_accent(tmp_path: Path) -> None:
    none = neural_status(None)
    assert "No NVIDIA GPU" in none
    assert "opt-in download" in none
    assert "accent" in none.lower()
    small = neural_status(GpuInfo("GTX 1050", 2048))
    assert str(MIN_VRAM_MIB) in small
    runtime = probe_runtime(
        tmp_path, gpu=BIG_GPU, torch_state=lambda: (False, False, "PyTorch missing")
    )
    with_runtime = neural_status(BIG_GPU, runtime)
    assert "enough" in with_runtime
    assert "not downloaded" in with_runtime
    assert "PyTorch missing" in with_runtime
    assert "Neural Engine is off" in with_runtime


# --- designer fallback ----------------------------------------------------


def test_neural_designer_unavailable_and_design_falls_back(tmp_path: Path) -> None:
    ok, reason = NeuralDesigner().available()
    assert not ok and "not installed" in reason
    off = probe_runtime(tmp_path, detect=lambda: None)
    ok, reason = NeuralDesigner(off).available()
    assert not ok and "no NVIDIA GPU" in reason
    with pytest.raises(RuntimeError):
        NeuralDesigner(off).design("anything")
    design = design_voice("a whispering assassin", runtime=off)
    assert design.designer == "lexicon"
    assert design.engine_id == DSP_ENGINE_ID


def test_neural_designer_makes_a_reference_clip_when_ready(tmp_path: Path) -> None:
    root = pack_root(tmp_path)
    _installed_pack(root, CONVERSION_PACK)
    _installed_pack(root, VOICE_DESIGN_PACK)
    runtime = probe_runtime(
        tmp_path, gpu=BIG_GPU, torch_state=lambda: (True, True, "ok")
    )

    class FakeMaker:
        def make_clip(self, instruct: str, text: str = "") -> tuple[np.ndarray, int]:
            t = np.arange(24000) / 24000
            return (0.2 * np.sin(2 * np.pi * 200 * t)).astype(np.float32), 24000

    designer = NeuralDesigner(runtime, clip_maker_factory=lambda _root: FakeMaker())
    assert designer.available() == (True, "")
    design = designer.design("a weary old ferryman with a voice like gravel")
    assert design.engine_id == NEURAL_ENGINE_ID
    clip = Path(design.params[REFERENCE_CLIP_KEY])
    assert clip.is_file() and clip.parent == tmp_path / "clips"
    from votr.clips import REFERENCE_CLIP_ID_KEY, ClipLibrary

    stored = ClipLibrary(tmp_path).get(design.params[REFERENCE_CLIP_ID_KEY])
    assert stored is not None and stored.origin == "designed"
    assert stored.name == "Weary Old Ferryman"
    assert design.params["mix"] == 1.0
    assert design.params["quality"] == 1.0
    assert design.name == "Weary Old Ferryman"
    assert any("Accent is still yours" in note for note in design.notes)


def test_design_voice_falls_back_when_the_neural_backend_blows_up(
    tmp_path: Path,
) -> None:
    root = pack_root(tmp_path)
    _installed_pack(root, CONVERSION_PACK)
    _installed_pack(root, VOICE_DESIGN_PACK)
    runtime = probe_runtime(
        tmp_path, gpu=BIG_GPU, torch_state=lambda: (True, True, "ok")
    )
    # Packs "present" but no torch/qwen_tts in this Python: backend import fails.
    design = design_voice("a whispering assassin", runtime=runtime)
    assert design.designer == "lexicon"
    assert design.engine_id == DSP_ENGINE_ID
    assert "whisper" in design.tone_tags
    assert any("neural designer failed" in note for note in design.notes)


# --- Session registration -------------------------------------------------


@pytest.mark.skipif(not stretch_available(), reason="DSP Engine needs python-stretch")
def test_session_without_gpu_keeps_neural_voices_disabled(tmp_path: Path) -> None:
    session = Session(tmp_path)
    assert NEURAL_ENGINE_ID not in INSTALLED_ENGINE_IDS
    assert session.neural.ready is False
    assert session.engine_installed(DSP_ENGINE_ID)
    assert not session.engine_installed(NEURAL_ENGINE_ID)
    assert session.ensure_neural_engine() is None
    assert session.should_preload_neural() is False
    session.draft.engine_id = NEURAL_ENGINE_ID
    assert session.preview_engine() is session.engine
    session.refresh_neural()
    assert session.neural_engine is None


@pytest.mark.skipif(not stretch_available(), reason="DSP Engine needs python-stretch")
def test_session_uses_neural_engine_for_neural_drafts_when_ready(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = Session(tmp_path)
    root = pack_root(tmp_path)
    _installed_pack(root, CONVERSION_PACK)
    session.neural = probe_runtime(
        tmp_path, gpu=BIG_GPU, torch_state=lambda: (True, True, "ok")
    )
    fake = FakeConverter()
    import votr.session as session_module

    monkeypatch.setattr(
        session_module,
        "build_neural_engine",
        lambda runtime, block_size=512: NeuralEngine(fake, block_size=block_size),
    )
    assert session.engine_installed(NEURAL_ENGINE_ID)
    assert session.should_preload_neural()
    clip = tmp_path / "ref.wav"
    write_wav(clip, (0.1 * np.sin(np.arange(48000) / 20.0)).astype(np.float32), 48000)
    session.draft.engine_id = NEURAL_ENGINE_ID
    session.draft.params = {REFERENCE_CLIP_KEY: str(clip), "mix": 1.0}
    engine = session.preview_engine()
    assert isinstance(engine, NeuralEngine)
    assert engine.has_reference and fake.reference == (48000, 48000)
    assert engine.block_size == session.engine.block_size
    session.draft.engine_id = DSP_ENGINE_ID
    assert session.preview_engine() is session.engine


def test_runtime_check_names_missing_packages_and_real_import_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from votr import neural_backends

    missing = neural_backends.missing_runtime_packages()
    # On this VM at least wandb/tensorboard are absent; the hint must name pips.
    assert missing and "wandb" in missing
    with pytest.raises(RuntimeError, match="pip install"):
        neural_backends.check_xvc_runtime(tmp_path)

    monkeypatch.setattr(neural_backends, "missing_runtime_packages", lambda: [])
    with pytest.raises(RuntimeError, match="source not found"):
        neural_backends.check_xvc_runtime(tmp_path)
    model_py = tmp_path / "models" / "codec" / "sac" / "model.py"
    model_py.parent.mkdir(parents=True)
    model_py.write_text("import definitely_not_a_module\n")

    def boom(_name):
        raise ImportError("No module named 'definitely_not_a_module'")

    monkeypatch.setattr(neural_backends.importlib, "import_module", boom)
    with pytest.raises(RuntimeError, match="definitely_not_a_module"):
        neural_backends.check_xvc_runtime(tmp_path)


@pytest.mark.skipif(not stretch_available(), reason="DSP Engine needs python-stretch")
def test_session_reports_why_the_engine_failed_to_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import votr.session as session_module

    session = Session(tmp_path)
    _installed_pack(pack_root(tmp_path), CONVERSION_PACK)
    session.neural = probe_runtime(
        tmp_path, gpu=BIG_GPU, torch_state=lambda: (True, True, "ok")
    )

    def failing(runtime, block_size=512):
        raise neural.NeuralUnavailable("X-VC needs Python packages: wandb tensorboard")

    monkeypatch.setattr(session_module, "build_neural_engine", failing)
    assert session.ensure_neural_engine() is None
    assert "wandb" in session.neural_error
    session.draft.engine_id = NEURAL_ENGINE_ID
    assert session.preview_engine() is session.engine
    session.refresh_neural()
    assert session.neural_error == ""


def test_audiotools_stub_is_constructible_like_xvc_needs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Michael's regression: hydra builds XVC with loss_config, which constructs
    STFTParams; a raise-on-construct dummy killed inference."""
    import importlib
    import sys

    from votr import neural_backends

    monkeypatch.delitem(sys.modules, "audiotools", raising=False)
    monkeypatch.setattr(neural_backends.importlib.util, "find_spec", lambda name: None)
    assert neural_backends.ensure_audiotools_stub() is True
    module = importlib.import_module("audiotools")
    assert getattr(module, "__votr_stub__", False)
    from audiotools import AudioSignal, STFTParams  # type: ignore[import-not-found]

    # Exactly what MelSpectrogramLoss.__init__ does, for each window length.
    params = [
        STFTParams(window_length=w, hop_length=w // 4, match_stride=False)
        for w in (2048, 512)
    ]
    assert params[0].window_length == 2048 and params[1].hop_length == 128
    assert "STFTParams" in repr(params[0])
    # AudioSignal can be built (loss forward does that); DSP on it cannot run.
    signal = AudioSignal(np.zeros(10, np.float32), 16000)
    assert signal.sample_rate == 16000
    with pytest.raises(RuntimeError, match="training"):
        signal.mel_spectrogram(80)
    # Second call is a no-op; an installed real package is never shadowed.
    assert neural_backends.ensure_audiotools_stub() is False
    monkeypatch.delitem(sys.modules, "audiotools", raising=False)
    monkeypatch.setattr(
        neural_backends.importlib.util, "find_spec", lambda name: object()
    )
    assert neural_backends.ensure_audiotools_stub() is False
    assert "audiotools" not in sys.modules


def test_xvc_target_constructs_with_loss_config_under_the_stub(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Replays hydra's instantiate(cfg.model.generator) against the stub with a
    faithful copy of X-VC's loss-building code path."""
    import sys

    from votr import neural_backends

    monkeypatch.delitem(sys.modules, "audiotools", raising=False)
    monkeypatch.setattr(neural_backends.importlib.util, "find_spec", lambda name: None)
    monkeypatch.setattr(neural_backends, "missing_runtime_packages", lambda: [])
    pkg = tmp_path / "models" / "codec" / "sac"
    (pkg / "blocks").mkdir(parents=True)
    for folder in (
        tmp_path / "models",
        tmp_path / "models" / "codec",
        pkg,
        pkg / "blocks",
    ):
        (folder / "__init__.py").write_text("")
    (pkg / "blocks" / "loss.py").write_text(
        "from audiotools import AudioSignal, STFTParams\n"
        "class MelSpectrogramLoss:\n"
        "    def __init__(self, window_lengths=(2048, 512), "
        "match_stride=False, **kw):\n"
        "        self.stft_params = [STFTParams(window_length=w, hop_length=w // 4,"
        " match_stride=match_stride) for w in window_lengths]\n"
    )
    (pkg / "model.py").write_text(
        "from audiotools import AudioSignal\n"
        "class XVC:\n"
        "    def __init__(self, loss_config=None, **kwargs):\n"
        "        self.loss_config = loss_config\n"
        "        if loss_config is not None:\n"
        "            from models.codec.sac.blocks import loss as losses\n"
        "            self.compute_mel_loss = losses.MelSpectrogramLoss("
        "**loss_config['mel_loss'])\n"
    )
    for name in [m for m in sys.modules if m == "models" or m.startswith("models.")]:
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.syspath_prepend(str(tmp_path))
    neural_backends.check_xvc_runtime(tmp_path)  # must not raise
    import models.codec.sac.model as xvc_model  # type: ignore[import-not-found]

    built = xvc_model.XVC(loss_config={"mel_loss": {"window_lengths": [2048, 512]}})
    assert len(built.compute_mel_loss.stft_params) == 2
    xvc_model.XVC(loss_config=None)


def test_patch_xvc_config_points_at_pack_and_drops_loss_config(tmp_path: Path) -> None:
    from votr.neural_backends import patch_xvc_config

    cfg = {
        "config": {
            "model": {
                "generator": {
                    "_target_": "models.codec.sac.model.XVC",
                    "loss_config": {"mel_loss": {"window_lengths": [2048, 512]}},
                    "semantic_encoder": {
                        "encoder": {
                            "from_pretrained": {"hf_repo": "x", "local_ckpt": None}
                        },
                        "cfg": {"hf_repo": "x", "local_ckpt": None},
                    },
                    "speaker_encoder": {"pretrained_dir": "pretrained/eres2net"},
                }
            }
        }
    }
    patched = patch_xvc_config(cfg, tmp_path)["config"]["model"]["generator"]
    assert patched["loss_config"] is None
    tokenizer = str(tmp_path / "glm-4-voice-tokenizer")
    assert (
        patched["semantic_encoder"]["encoder"]["from_pretrained"]["local_ckpt"]
        == tokenizer
    )
    assert patched["semantic_encoder"]["cfg"]["local_ckpt"] == tokenizer
    assert patched["speaker_encoder"]["pretrained_dir"].endswith(
        "speech_eres2net_sv_en_voxceleb_16k"
    )
    flat = {
        "model": {
            "generator": {
                "semantic_encoder": {"encoder": {"from_pretrained": {}}, "cfg": {}},
                "speaker_encoder": {},
            }
        }
    }
    assert "loss_config" not in patch_xvc_config(flat, tmp_path)["model"]["generator"]


def test_neural_extra_has_no_protobuf_fight() -> None:
    import tomllib

    data = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    extra = data["project"]["optional-dependencies"]["neural"]
    joined = " ".join(extra).lower()
    assert "descript-audiotools" not in joined
    assert "protobuf" not in joined
    assert "audiotools" not in [
        m
        for m, _ in __import__(
            "votr.neural_backends", fromlist=["XVC_RUNTIME_MODULES"]
        ).XVC_RUNTIME_MODULES
    ]

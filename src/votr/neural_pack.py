# Copyright (C) 2026 Michael Cobb
# SPDX-License-Identifier: GPL-3.0-or-later

"""Neural model packs: what they are, where they live, and an opt-in downloader.

Nothing here runs unless the GM presses Download in Settings → Neural. Files
come straight from their publishers (Hugging Face, ModelScope, GitHub) into the
user-data folder; the app never redistributes weights. Every file records its
size, licence and (where the publisher exposes one) SHA-256, and the download
resumes, verifies and can be cancelled.

Facts below were read from the publishers on 2026-09-20; sizes are bytes.
"""

from __future__ import annotations

import hashlib
import shutil
import threading
import zipfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

PACK_DIR_NAME = "neural"
XVC_COMMIT = "49df8c591eafc48b096e466d96f9839f9c0dd739"
GIB = 1024**3


class DownloadCancelled(Exception):
    """The GM pressed Cancel; partial files are kept for resume."""


class DownloadError(Exception):
    """A file could not be fetched or failed its checksum."""


@dataclass(frozen=True)
class PackFile:
    relpath: str
    url: str
    size: int | None
    sha256: str | None
    license_id: str
    unzip_to: str | None = None


@dataclass(frozen=True)
class ModelPack:
    pack_id: str
    title: str
    summary: str
    files: tuple[PackFile, ...]
    licences: tuple[str, ...]
    caveats: tuple[str, ...] = ()

    @property
    def total_bytes(self) -> int:
        return sum(item.size or 0 for item in self.files)

    @property
    def total_label(self) -> str:
        return f"{self.total_bytes / GIB:.1f} GB"


def _hf(repo: str, filename: str) -> str:
    return f"https://huggingface.co/{repo}/resolve/main/{filename}"


_GLM_LICENCE = (
    "GLM-4-Voice tokenizer (zai-org/glm-4-voice-tokenizer): custom glm-4-voice "
    "licence — free for personal and research use; commercial use requires "
    "registration with Zhipu AI; products must show “Built with glm-4”; "
    "governed by PRC law. Not an OSI licence. X-VC cannot run without it."
)

CONVERSION_PACK = ModelPack(
    pack_id="xvc",
    title="X-VC zero-shot voice conversion",
    summary=(
        "Sound like a different person from a reference clip (10–30 s of "
        "clear speech is better; 30 s max). 16 kHz streaming model; NVIDIA "
        "GPU with about 6 GB free memory."
    ),
    files=(
        PackFile(
            relpath="xvc-src.zip",
            url=f"https://github.com/Jerrister/X-VC/archive/{XVC_COMMIT}.zip",
            size=None,
            sha256=None,
            license_id="MIT",
            unzip_to="xvc-src",
        ),
        PackFile(
            relpath="xvc/xvc.pt",
            url=_hf("chenxie95/X-VC", "xvc.pt"),
            size=5_007_756_915,
            sha256="1ba0ca3187d2a6753a1529db18c5490e5cb20c8874dc067b92935ff39cfed687",
            license_id="MIT",
        ),
        PackFile(
            relpath="xvc/config.json",
            url=_hf("chenxie95/X-VC", "config.json"),
            size=404,
            sha256=None,
            license_id="MIT",
        ),
        PackFile(
            relpath="glm-4-voice-tokenizer/model.safetensors",
            url=_hf("zai-org/glm-4-voice-tokenizer", "model.safetensors"),
            size=1_458_374_480,
            sha256="2800bd503f52b51e45f0c53cfd5c31dcfe8ef7f13d22b396aa3d53e0280dd1e4",
            license_id="glm-4-voice",
        ),
        PackFile(
            relpath="glm-4-voice-tokenizer/config.json",
            url=_hf("zai-org/glm-4-voice-tokenizer", "config.json"),
            size=1746,
            sha256=None,
            license_id="glm-4-voice",
        ),
        PackFile(
            relpath="glm-4-voice-tokenizer/preprocessor_config.json",
            url=_hf("zai-org/glm-4-voice-tokenizer", "preprocessor_config.json"),
            size=340,
            sha256=None,
            license_id="glm-4-voice",
        ),
        PackFile(
            relpath="glm-4-voice-tokenizer/LICENSE",
            url=_hf("zai-org/glm-4-voice-tokenizer", "LICENSE"),
            size=6490,
            sha256=None,
            license_id="glm-4-voice",
        ),
        PackFile(
            relpath="speech_eres2net_sv_en_voxceleb_16k/pretrained_eres2net.ckpt",
            url=(
                "https://modelscope.cn/models/iic/speech_eres2net_sv_en_voxceleb_16k"
                "/resolve/master/pretrained_eres2net.ckpt"
            ),
            size=26_725_867,
            sha256="d8941f5952e31820173c8854562cb6d7897aaa58cd65c18f30d5a2e52d30847d",
            license_id="Apache-2.0",
        ),
        PackFile(
            relpath="speech_eres2net_sv_en_voxceleb_16k/configuration.json",
            url=(
                "https://modelscope.cn/models/iic/speech_eres2net_sv_en_voxceleb_16k"
                "/resolve/master/configuration.json"
            ),
            size=405,
            sha256="bbd0c639f5c73325ad9f080d9f8866b613f739546216dc090eabe65ed0bbdd18",
            license_id="Apache-2.0",
        ),
    ),
    licences=(
        "X-VC code and checkpoint (Jerrister/X-VC, chenxie95/X-VC): MIT.",
        _GLM_LICENCE,
        "ERes2Net speaker encoder (iic/speech_eres2net_sv_en_voxceleb_16k): "
        "Apache-2.0.",
    ),
    caveats=(
        "Also needs CUDA PyTorch and X-VC's Python dependencies, which this "
        'downloader does not install: pip install -e ".[neural]" (see '
        "docs/neural-voice.md).",
        "Verified converting speech on Windows 11 / Python 3.13 / torch 2.9 / "
        "transformers 4.46 (2026-09-20).",
    ),
)

VOICE_DESIGN_PACK = ModelPack(
    pack_id="qwen3-tts-voicedesign",
    title="Qwen3-TTS VoiceDesign (prompt → reference clip)",
    summary=(
        "Turns Tone Hints into a spoken reference clip that the conversion model "
        "then imitates. 1.7 B parameters; NVIDIA GPU."
    ),
    files=tuple(
        PackFile(
            relpath=f"qwen3-tts-voicedesign/{name}",
            url=_hf("Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign", name),
            size=size,
            sha256=sha,
            license_id="Apache-2.0",
        )
        for name, size, sha in (
            ("config.json", 4421, None),
            ("generation_config.json", 245, None),
            ("merges.txt", 1_671_839, None),
            (
                "model.safetensors",
                3_833_402_552,
                "391e8db219f292c515297cdceeb43e4eae67cdde35fa57e79a6a8a532fca0522",
            ),
            ("preprocessor_config.json", 127, None),
            ("speech_tokenizer/config.json", 2336, None),
            ("speech_tokenizer/configuration.json", 76, None),
            (
                "speech_tokenizer/model.safetensors",
                682_293_092,
                "836b7b357f5ea43e889936a3709af68dfe3751881acefe4ecf0dbd30ba571258",
            ),
            ("speech_tokenizer/preprocessor_config.json", 234, None),
            ("tokenizer_config.json", 7344, None),
            ("vocab.json", 2_776_833, None),
        )
    ),
    licences=("Qwen3-TTS-12Hz-1.7B-VoiceDesign (Qwen): Apache-2.0.",),
    caveats=(
        "Also needs the `qwen-tts` Python package (Apache-2.0) and CUDA PyTorch.",
        "Verified generating a reference clip on Windows 11 / Python 3.13 / "
        "torch 2.9 (2026-09-20).",
    ),
)

PACKS: tuple[ModelPack, ...] = (CONVERSION_PACK, VOICE_DESIGN_PACK)


def pack_root(data_dir: Path) -> Path:
    return Path(data_dir) / PACK_DIR_NAME


def pack_by_id(pack_id: str) -> ModelPack | None:
    return next((pack for pack in PACKS if pack.pack_id == pack_id), None)


@dataclass
class PackStatus:
    pack: ModelPack
    present: list[PackFile] = field(default_factory=list)
    missing: list[PackFile] = field(default_factory=list)

    @property
    def installed(self) -> bool:
        return not self.missing

    @property
    def bytes_present(self) -> int:
        return sum(item.size or 0 for item in self.present)


def _final_path(root: Path, item: PackFile) -> Path:
    if item.unzip_to:
        return root / item.unzip_to
    return root / item.relpath


def _is_present(root: Path, item: PackFile) -> bool:
    target = _final_path(root, item)
    if item.unzip_to:
        return target.is_dir() and any(target.iterdir())
    if not target.is_file():
        return False
    if item.size is not None and target.stat().st_size != item.size:
        return False
    return True


def pack_status(pack: ModelPack, root: Path) -> PackStatus:
    status = PackStatus(pack)
    for item in pack.files:
        (status.present if _is_present(root, item) else status.missing).append(item)
    return status


@dataclass(frozen=True)
class DownloadProgress:
    pack_id: str
    file_relpath: str
    file_index: int
    file_count: int
    bytes_done: int
    bytes_total: int | None


Opener = Callable[[Request], object]


def _default_opener(request: Request):
    return urlopen(request, timeout=60)


def _sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fetch_file(
    item: PackFile,
    destination: Path,
    *,
    opener: Opener,
    cancel: threading.Event | None,
    report: Callable[[int, int | None], None],
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    done = partial.stat().st_size if partial.exists() else 0
    if item.size is not None and done > item.size:
        partial.unlink()
        done = 0
    headers = {"User-Agent": "VoiceOfTheRealm/neural-pack"}
    if done:
        headers["Range"] = f"bytes={done}-"
    request = Request(item.url, headers=headers)
    try:
        response = opener(request)
    except HTTPError as exc:
        if exc.code == 416 and done and (item.size is None or done == item.size):
            response = None
        else:
            raise DownloadError(f"{item.relpath}: HTTP {exc.code}") from exc
    except OSError as exc:
        raise DownloadError(f"{item.relpath}: {exc}") from exc
    if response is not None:
        status = getattr(response, "status", 200)
        mode = "ab" if (done and status == 206) else "wb"
        if mode == "wb":
            done = 0
        total = item.size
        if total is None:
            length = response.headers.get("Content-Length")
            if length and length.isdigit():
                total = int(length) + (done if status == 206 else 0)
        with partial.open(mode) as handle:
            report(done, total)
            while True:
                if cancel is not None and cancel.is_set():
                    raise DownloadCancelled(item.relpath)
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                handle.write(chunk)
                done += len(chunk)
                report(done, total)
        if hasattr(response, "close"):
            response.close()
    if item.size is not None and partial.stat().st_size != item.size:
        raise DownloadError(
            f"{item.relpath}: got {partial.stat().st_size} bytes, expected {item.size}"
        )
    if item.sha256 is not None and _sha256_of(partial) != item.sha256.lower():
        partial.unlink()
        raise DownloadError(f"{item.relpath}: checksum mismatch — removed, retry")
    partial.replace(destination)


def download_pack(
    pack: ModelPack,
    root: Path,
    *,
    progress: Callable[[DownloadProgress], None] | None = None,
    cancel: threading.Event | None = None,
    opener: Opener = _default_opener,
) -> PackStatus:
    """Fetch every missing file of ``pack`` into ``root``; resume, verify, unzip.

    Raises ``DownloadCancelled`` when ``cancel`` is set (partial files stay for a
    later resume) and ``DownloadError`` on HTTP, size or checksum failure.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    status = pack_status(pack, root)
    count = len(status.missing)
    for index, item in enumerate(list(status.missing)):

        def report(done: int, total: int | None, _item=item, _index=index) -> None:
            if progress is not None:
                progress(
                    DownloadProgress(
                        pack.pack_id, _item.relpath, _index, count, done, total
                    )
                )

        if item.unzip_to:
            archive = root / item.relpath
            if not archive.is_file():
                _fetch_file(item, archive, opener=opener, cancel=cancel, report=report)
            _unzip_flat(archive, root / item.unzip_to)
            archive.unlink(missing_ok=True)
        else:
            _fetch_file(
                item, root / item.relpath, opener=opener, cancel=cancel, report=report
            )
    return pack_status(pack, root)


def _unzip_flat(archive: Path, target: Path) -> None:
    """Unzip, dropping the single top-level folder GitHub archives carry."""
    if target.exists():
        shutil.rmtree(target)
    with zipfile.ZipFile(archive) as bundle:
        names = [name for name in bundle.namelist() if not name.endswith("/")]
        roots = {name.split("/", 1)[0] for name in names}
        strip = len(roots) == 1 and all("/" in name for name in names)
        for name in names:
            relative = name.split("/", 1)[1] if strip else name
            if not relative or ".." in Path(relative).parts:
                continue
            out = target / relative
            out.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(name) as src, out.open("wb") as dst:
                shutil.copyfileobj(src, dst)


def remove_pack(pack: ModelPack, root: Path) -> None:
    for item in pack.files:
        path = _final_path(root, item)
        if item.unzip_to and path.is_dir():
            shutil.rmtree(path)
        elif path.is_file():
            path.unlink()
        partial = (root / item.relpath).with_suffix(Path(item.relpath).suffix + ".part")
        if partial.is_file():
            partial.unlink()

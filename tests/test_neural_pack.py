# Copyright (C) 2026 Michael Cobb
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import hashlib
import io
import re
import threading
import zipfile
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from votr.neural_pack import (
    CONVERSION_PACK,
    PACKS,
    VOICE_DESIGN_PACK,
    DownloadCancelled,
    DownloadError,
    ModelPack,
    PackFile,
    download_pack,
    pack_by_id,
    pack_status,
    remove_pack,
)


class _RangeServer(threading.Thread):
    """Tiny HTTP server serving bytes with Range support; counts requests."""

    def __init__(self, files: dict[str, bytes], *, chunked: set[str] = frozenset()):
        super().__init__(daemon=True)
        self.files = files
        self.chunked = chunked
        self.requests: list[tuple[str, str | None]] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args) -> None:
                return None

            def do_GET(self) -> None:  # noqa: N802
                name = self.path.lstrip("/")
                body = outer.files.get(name)
                outer.requests.append((name, self.headers.get("Range")))
                if body is None:
                    self.send_response(404)
                    self.end_headers()
                    return
                start = 0
                header = self.headers.get("Range")
                if header:
                    match = re.match(r"bytes=(\d+)-", header)
                    start = int(match.group(1)) if match else 0
                    if start >= len(body):
                        self.send_response(416)
                        self.end_headers()
                        return
                    self.send_response(206)
                else:
                    self.send_response(200)
                payload = body[start:]
                if name in outer.chunked:
                    self.send_header("Transfer-Encoding", "chunked")
                    self.end_headers()
                    self.wfile.write(
                        f"{len(payload):x}\r\n".encode() + payload + b"\r\n0\r\n\r\n"
                    )
                    return
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}/"

    def run(self) -> None:
        self.server.serve_forever()

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def server():
    payload = bytes(range(256)) * 4096  # 1 MiB
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as bundle:
        bundle.writestr("X-VC-abc123/README.md", "hello")
        bundle.writestr("X-VC-abc123/bins/infer_utils.py", "print(1)")
    files = {"model.bin": payload, "small.json": b"{}", "src.zip": buffer.getvalue()}
    srv = _RangeServer(files, chunked={"src.zip"})
    srv.start()
    yield srv, payload
    srv.stop()


def _pack(
    url: str, payload: bytes, *, sha: str | None = None, size: int | None = None
) -> ModelPack:
    return ModelPack(
        pack_id="test",
        title="Test pack",
        summary="",
        files=(
            PackFile("src.zip", url + "src.zip", None, None, "MIT", unzip_to="src"),
            PackFile(
                "weights/model.bin",
                url + "model.bin",
                len(payload) if size is None else size,
                hashlib.sha256(payload).hexdigest() if sha is None else sha,
                "MIT",
            ),
            PackFile("weights/small.json", url + "small.json", 2, None, "MIT"),
        ),
        licences=("MIT",),
    )


def test_manifest_facts_are_well_formed() -> None:
    assert pack_by_id("xvc") is CONVERSION_PACK
    assert pack_by_id("qwen3-tts-voicedesign") is VOICE_DESIGN_PACK
    assert pack_by_id("nope") is None
    for pack in PACKS:
        assert pack.licences and pack.summary
        assert pack.total_bytes > 1024**3
        seen = set()
        for item in pack.files:
            assert item.relpath not in seen
            seen.add(item.relpath)
            assert item.url.startswith("https://")
            assert item.license_id
            if item.sha256 is not None:
                assert re.fullmatch(r"[0-9a-f]{64}", item.sha256)
            if item.unzip_to is None:
                assert item.size is not None and item.size > 0
    # Big checkpoints carry checksums; licences are spelled out, GLM caveat included.
    big = [item for pack in PACKS for item in pack.files if (item.size or 0) > 10**8]
    assert big and all(item.sha256 for item in big)
    assert any("glm-4-voice" in text for text in CONVERSION_PACK.licences)
    assert "GB" in CONVERSION_PACK.total_label
    assert CONVERSION_PACK.total_bytes // 10**9 == 6


def test_download_installs_verifies_and_unzips(tmp_path: Path, server) -> None:
    srv, payload = server
    pack = _pack(srv.url, payload)
    seen = []
    status = download_pack(pack, tmp_path, progress=seen.append)
    assert status.installed
    assert (tmp_path / "weights" / "model.bin").read_bytes() == payload
    assert (tmp_path / "src" / "bins" / "infer_utils.py").read_text() == "print(1)"
    assert not (tmp_path / "src.zip").exists()
    assert not list(tmp_path.rglob("*.part"))
    assert (
        seen[-1].bytes_done == 2
        and seen[-1].file_index == 2
        and seen[-1].file_count == 3
    )
    assert any(
        p.file_relpath == "weights/model.bin" and p.bytes_total == len(payload)
        for p in seen
    )
    # Second call is a no-op: nothing missing, no requests made.
    before = len(srv.requests)
    assert download_pack(pack, tmp_path).installed
    assert len(srv.requests) == before


def test_download_resumes_a_partial_file(tmp_path: Path, server) -> None:
    srv, payload = server
    pack = _pack(srv.url, payload)
    partial = tmp_path / "weights" / "model.bin.part"
    partial.parent.mkdir(parents=True)
    partial.write_bytes(payload[:300_000])
    download_pack(pack, tmp_path)
    ranges = [rng for name, rng in srv.requests if name == "model.bin"]
    assert ranges == ["bytes=300000-"]
    assert (tmp_path / "weights" / "model.bin").read_bytes() == payload


def test_cancel_keeps_partial_and_reports(tmp_path: Path, server) -> None:
    srv, payload = server
    pack = _pack(srv.url, payload)
    cancel = threading.Event()

    def on_progress(progress) -> None:
        if progress.file_relpath == "weights/model.bin" and progress.bytes_done > 0:
            cancel.set()

    with pytest.raises(DownloadCancelled):
        download_pack(pack, tmp_path, progress=on_progress, cancel=cancel)
    status = pack_status(pack, tmp_path)
    assert not status.installed
    assert (tmp_path / "weights" / "model.bin.part").exists()
    assert not (tmp_path / "weights" / "model.bin").exists()


def test_checksum_and_size_mismatches_fail_loudly(tmp_path: Path, server) -> None:
    srv, payload = server
    bad_sha = _pack(srv.url, payload, sha="0" * 64)
    with pytest.raises(DownloadError, match="checksum"):
        download_pack(bad_sha, tmp_path)
    assert not (tmp_path / "weights" / "model.bin").exists()
    assert not (tmp_path / "weights" / "model.bin.part").exists()
    bad_size = _pack(srv.url, payload, size=len(payload) + 5)
    with pytest.raises(DownloadError, match="expected"):
        download_pack(bad_size, tmp_path)


def test_missing_file_is_a_download_error(tmp_path: Path, server) -> None:
    srv, payload = server
    pack = ModelPack(
        "x",
        "x",
        "",
        (PackFile("gone.bin", srv.url + "gone.bin", 3, None, "MIT"),),
        ("MIT",),
    )
    with pytest.raises(DownloadError, match="HTTP 404"):
        download_pack(pack, tmp_path)


def test_status_and_remove(tmp_path: Path, server) -> None:
    srv, payload = server
    pack = _pack(srv.url, payload)
    assert not pack_status(pack, tmp_path).installed
    download_pack(pack, tmp_path)
    assert pack_status(pack, tmp_path).installed
    # A truncated file counts as missing again.
    (tmp_path / "weights" / "model.bin").write_bytes(payload[:10])
    status = pack_status(pack, tmp_path)
    assert [item.relpath for item in status.missing] == ["weights/model.bin"]
    remove_pack(pack, tmp_path)
    assert not (tmp_path / "src").exists()
    assert not (tmp_path / "weights" / "small.json").exists()

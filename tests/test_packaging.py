# Copyright (C) 2026 Michael Cobb
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from read_version import read_version  # noqa: E402


def test_version_matches_pyproject() -> None:
    from votr import __version__

    assert read_version() == __version__ == "0.1.0"


def test_inno_is_per_user_and_shows_gpl() -> None:
    text = (ROOT / "installer" / "votr.iss").read_text(encoding="utf-8")
    assert "PrivilegesRequired=lowest" in text
    assert "LicenseFile=..\\LICENSE" in text or "LicenseFile=../LICENSE" in text
    assert "THIRD_PARTY_NOTICES.md" in text
    assert "VB-CABLE" in text or "Virtual Cable" in text
    assert "{userprograms}" in text or "{group}" in text


def test_spec_is_onedir_without_collect_all() -> None:
    spec = (ROOT / "packaging" / "VoiceOfTheRealm.spec").read_text(encoding="utf-8")
    assert "COLLECT" in spec
    assert "collect-all" not in spec
    assert "votr/assets" in spec
    assert "sounddevice" in spec
    assert "pedalboard" in spec


def test_release_workflow_is_tag_triggered() -> None:
    workflow = (ROOT / ".github" / "workflows" / "release.yml").read_text(
        encoding="utf-8"
    )
    assert "v*" in workflow
    assert "windows-latest" in workflow
    assert "sha256" in workflow.lower()
    assert "innosetup" in workflow.lower()


def test_acceptance_doc_is_not_a_fake_run() -> None:
    text = (ROOT / "docs" / "spikes" / "acceptance-rough-draft.md").read_text(
        encoding="utf-8"
    )
    assert "not run" in text.lower()
    assert "Grimjaw" in text
    assert "never" in text.lower() or "only" in text.lower()


def test_stamp_inno_version(tmp_path: Path, monkeypatch) -> None:
    import stamp_inno_version as stamp

    monkeypatch.setattr(stamp, "OUT", tmp_path / "version.iss")
    assert stamp.main() == 0
    assert 'MyAppVersion "0.1.0"' in (tmp_path / "version.iss").read_text(
        encoding="utf-8"
    )


def test_changelog_mentions_installer() -> None:
    text = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert "0.1.0" in text
    assert "Inno" in text or "installer" in text.lower()

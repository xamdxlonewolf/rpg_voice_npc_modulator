# Copyright (C) 2026 Michael Cobb
# SPDX-License-Identifier: GPL-3.0-or-later

"""One JSON file per Voice in the user-data folder."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from votr.voice import Voice

log = logging.getLogger("votr.store")


def default_data_dir() -> Path:
    import platformdirs

    return Path(platformdirs.user_data_dir("VoiceOfTheRealm"))


class VoiceStore:
    def __init__(self, root: Path | None = None) -> None:
        base = root if root is not None else default_data_dir()
        self.root = Path(base) / "voices"
        self.root.mkdir(parents=True, exist_ok=True)
        self.warnings: list[str] = []

    def path_for(self, voice_id: str) -> Path:
        return self.root / f"{voice_id}.json"

    def load_all(self) -> list[Voice]:
        self.warnings = []
        voices: list[Voice] = []
        for path in sorted(self.root.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(data, dict):
                    raise ValueError("root must be an object")
                voices.append(Voice.from_dict(data))
            except (OSError, json.JSONDecodeError, ValueError, TypeError) as exc:
                message = f"Skipped corrupt Voice file {path.name}: {exc}"
                self.warnings.append(message)
                log.warning(message)
        return voices

    def save(self, voice: Voice) -> Path:
        voice.touch()
        path = self.path_for(voice.id)
        path.write_text(
            json.dumps(voice.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return path

    def delete(self, voice_id: str) -> None:
        path = self.path_for(voice_id)
        if path.exists():
            path.unlink()

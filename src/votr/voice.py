# Copyright (C) 2026 Michael Cobb
# SPDX-License-Identifier: GPL-3.0-or-later

"""Voice model: one named character sound targeting one Engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

DSP_ENGINE_ID = "dsp-v1"

_KNOWN = (
    "id",
    "name",
    "engine_id",
    "tone_hints",
    "tone_tags",
    "params",
    "colour",
    "icon",
    "preset",
    "created",
    "updated",
    "last_used",
)


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


@dataclass
class Voice:
    id: str
    name: str
    engine_id: str = DSP_ENGINE_ID
    tone_hints: str = ""
    tone_tags: list[str] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)
    colour: str = "#5c4d7a"
    icon: str = ""
    preset: str = ""
    created: str = ""
    updated: str = ""
    last_used: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def new(cls, name: str = "New Voice") -> Voice:
        stamp = _now()
        return cls(
            id=str(uuid4()),
            name=name,
            created=stamp,
            updated=stamp,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Voice:
        if "id" not in data or "name" not in data:
            raise ValueError("Voice JSON requires id and name")
        extra = {key: value for key, value in data.items() if key not in _KNOWN}
        tags = data.get("tone_tags") or []
        params = data.get("params") or {}
        if not isinstance(tags, list) or not isinstance(params, dict):
            raise ValueError("tone_tags must be a list and params a dict")
        return cls(
            id=str(data["id"]),
            name=str(data["name"]),
            engine_id=str(data.get("engine_id") or DSP_ENGINE_ID),
            tone_hints=str(data.get("tone_hints") or ""),
            tone_tags=[str(tag) for tag in tags],
            params=dict(params),
            colour=str(data.get("colour") or "#5c4d7a"),
            icon=str(data.get("icon") or ""),
            preset=str(data.get("preset") or ""),
            created=str(data.get("created") or _now()),
            updated=str(data.get("updated") or _now()),
            last_used=str(data.get("last_used") or ""),
            extra=extra,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "id": self.id,
            "name": self.name,
            "engine_id": self.engine_id,
            "tone_hints": self.tone_hints,
            "tone_tags": list(self.tone_tags),
            "params": dict(self.params),
            "colour": self.colour,
            "icon": self.icon,
            "preset": self.preset,
            "created": self.created,
            "updated": self.updated,
            "last_used": self.last_used,
        }
        payload.update(self.extra)
        return payload

    def touch(self, *, used: bool = False) -> None:
        stamp = _now()
        self.updated = stamp
        if used:
            self.last_used = stamp

    def duplicate(self) -> Voice:
        copy = Voice.from_dict(self.to_dict())
        copy.id = str(uuid4())
        copy.name = f"{self.name} copy"
        copy.created = _now()
        copy.updated = copy.created
        copy.last_used = ""
        return copy

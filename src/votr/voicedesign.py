# Copyright (C) 2026 Michael Cobb
# SPDX-License-Identifier: GPL-3.0-or-later

"""Prompt → Voice: turn a free-text description into Tone Tags and sliders.

Two designers sit behind one small interface:

* ``LexiconDesigner`` — always available, CPU, offline, instant. It maps words
  in the prompt to Tone Tag Macros and slider nudges. It shapes *how the GM
  sounds*; it cannot invent a new person or an accent (see docs/neural-voice.md).
* ``NeuralDesigner`` (votr.neural) — the seam for a local voice-design model. It
  is not installed in this build and says so; ``design_voice`` falls back.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol

from votr.dsp import SCHEMA
from votr.macros import Macro, apply_macros, load_macros

_SCHEMA = {spec.key: spec for spec in SCHEMA}

# Words that pick a Tone Tag Macro (applied in prompt order; later tags win).
_TAG_WORDS: dict[str, tuple[str, ...]] = {
    "gravelly": (
        "gravel",
        "gravelly",
        "rough",
        "raspy",
        "hoarse",
        "gruff",
        "orc",
        "dwarf",
        "dwarven",
        "pirate",
        "smoker",
    ),
    "booming": (
        "booming",
        "boom",
        "thunder",
        "thunderous",
        "commanding",
        "warlord",
        "captain",
        "general",
    ),
    "frail": (
        "frail",
        "old",
        "ancient",
        "elderly",
        "aged",
        "crone",
        "hag",
        "weak",
        "dying",
        "feeble",
        "wizened",
    ),
    "hollow": (
        "hollow",
        "empty",
        "cavern",
        "cavernous",
        "cave",
        "tunnel",
        "barrel",
        "helmet",
        "visor",
        "mask",
        "tomb",
        "crypt",
    ),
    "tiny": (
        "tiny",
        "small",
        "little",
        "fairy",
        "pixie",
        "sprite",
        "gnome",
        "mouse",
        "squeaky",
        "kobold",
        "imp",
    ),
    "giant": (
        "giant",
        "huge",
        "massive",
        "enormous",
        "towering",
        "troll",
        "ogre",
        "colossal",
        "titan",
        "dragon",
        "demon",
        "devil",
        "fiend",
        "bear",
    ),
    "ghostly": (
        "ghost",
        "ghostly",
        "spirit",
        "spectral",
        "spectre",
        "specter",
        "wraith",
        "phantom",
        "undead",
        "lich",
        "ethereal",
        "haunting",
        "haunted",
        "banshee",
    ),
    "robotic": (
        "robot",
        "robotic",
        "machine",
        "mechanical",
        "automaton",
        "construct",
        "metallic",
        "clockwork",
        "golem",
        "android",
        "synthetic",
    ),
    "nasal": (
        "nasal",
        "twang",
        "twangy",
        "whiny",
        "whining",
        "goblin",
        "rat",
        "sniveling",
        "snivelling",
    ),
    "whisper": (
        "whisper",
        "whispering",
        "whispered",
        "hushed",
        "hush",
        "secretive",
        "conspiratorial",
        "assassin",
    ),
    "regal": (
        "regal",
        "noble",
        "king",
        "queen",
        "royal",
        "lord",
        "lady",
        "court",
        "herald",
        "emperor",
        "empress",
        "prince",
        "princess",
        "majestic",
    ),
    "sly": (
        "sly",
        "cunning",
        "sneaky",
        "thief",
        "rogue",
        "trickster",
        "smooth",
        "silky",
        "charming",
        "seductive",
    ),
}

# Words that nudge a slider directly (added after the Macros, then clipped).
_NUDGE_WORDS: dict[str, tuple[tuple[str, float], ...]] = {
    "deep": (("pitch_semitones", -3.0), ("formant_semitones", -1.5)),
    "low": (("pitch_semitones", -3.0),),
    "bass": (("pitch_semitones", -4.0), ("formant_semitones", -2.0)),
    "baritone": (("pitch_semitones", -2.0),),
    "high": (("pitch_semitones", 3.0),),
    "shrill": (("pitch_semitones", 4.0), ("tonality", -0.15)),
    "chesty": (("formant_semitones", -3.0),),
    "thin": (("formant_semitones", 3.0),),
    "young": (("pitch_semitones", 3.0), ("formant_semitones", 2.0)),
    "child": (("pitch_semitones", 5.0), ("formant_semitones", 4.0)),
    "kid": (("pitch_semitones", 5.0), ("formant_semitones", 4.0)),
    "boy": (("pitch_semitones", 5.0), ("formant_semitones", 4.0)),
    "girl": (("pitch_semitones", 6.0), ("formant_semitones", 4.0)),
    "woman": (("pitch_semitones", 4.0), ("formant_semitones", 3.0)),
    "female": (("pitch_semitones", 4.0), ("formant_semitones", 3.0)),
    "man": (("pitch_semitones", -2.0), ("formant_semitones", -1.0)),
    "male": (("pitch_semitones", -2.0), ("formant_semitones", -1.0)),
    "angry": (("growl", 0.3),),
    "furious": (("growl", 0.45),),
    "snarling": (("growl", 0.4),),
    "growling": (("growl", 0.4),),
    "growl": (("growl", 0.4),),
    "gruff": (("growl", 0.25),),
    "raspy": (("growl", 0.3),),
    "hoarse": (("growl", 0.25), ("breath", 0.15)),
    "gravelly": (("growl", 0.3),),
    "gravel": (("growl", 0.3),),
    "rough": (("growl", 0.25),),
    "smoky": (("growl", 0.2), ("tonality", 0.1)),
    "breathy": (("breath", 0.35),),
    "airy": (("breath", 0.3),),
    "sighing": (("breath", 0.3),),
    "bright": (("tonality", -0.2),),
    "clear": (("tonality", -0.15),),
    "crisp": (("tonality", -0.15),),
    "dark": (("tonality", 0.2),),
    "muffled": (("tonality", 0.25), ("distance", 0.2)),
    "warm": (("tonality", 0.12),),
    "dull": (("tonality", 0.2),),
    "distant": (("distance", 0.35),),
    "far": (("distance", 0.3),),
    "faraway": (("distance", 0.35),),
    "echo": (("room", 0.35),),
    "echoing": (("room", 0.4),),
    "hall": (("room", 0.3),),
    "cathedral": (("room", 0.5),),
    "temple": (("room", 0.4),),
    "chamber": (("room", 0.3),),
    "dungeon": (("room", 0.35), ("tonality", 0.1)),
    "dry": (("room", -0.3),),
    "close": (("distance", -0.2), ("room", -0.2)),
    "intimate": (("distance", -0.2), ("room", -0.2)),
    "quiet": (("gate", 0.2),),
}

_MORE = ("very", "extremely", "really", "deeply", "super", "incredibly")
_LESS = ("slightly", "somewhat", "bit", "little", "touch", "faintly", "mildly")

# Recognised but not actionable by the DSP Engine — kept as notes, never faked.
_ACCENT_WORDS = (
    "irish",
    "british",
    "english",
    "scottish",
    "scots",
    "welsh",
    "cockney",
    "posh",
    "french",
    "german",
    "russian",
    "italian",
    "spanish",
    "american",
    "southern",
    "australian",
    "texan",
    "yorkshire",
    "accent",
    "brogue",
    "dialect",
)
_IDENTITY_WORDS = ("sound like", "sounds like", "voice of", "impression of")

ACCENT_NOTE = (
    "Accent words were kept as Tone Hints only. The DSP Engine changes how you "
    "sound, not how you pronounce words — see docs/neural-voice.md."
)
IDENTITY_NOTE = (
    '"Sound like a specific person" needs the Neural Engine (NVIDIA GPU, not '
    "in this build). Sliders were set from the descriptive words only."
)
_STOP = {
    "a",
    "an",
    "the",
    "of",
    "with",
    "and",
    "who",
    "that",
    "very",
    "voice",
    "is",
    "in",
    "at",
    "on",
    "from",
    "for",
    "to",
    "by",
    "his",
    "her",
    "their",
    "my",
    "like",
}


@dataclass
class VoiceDesign:
    name: str
    tone_tags: list[str]
    params: dict[str, float]
    matched: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    designer: str = "lexicon"


class VoiceDesigner(Protocol):
    designer_id: str

    def available(self) -> tuple[bool, str]:
        """(True, '') when this designer can run here; else (False, why not)."""

    def design(self, prompt: str) -> VoiceDesign:
        """Turn a free-text description into tags and sliders."""


def _tokens(prompt: str) -> list[str]:
    return re.findall(r"[a-z][a-z'-]*", prompt.lower())


def _clip(key: str, value: float) -> float:
    spec = _SCHEMA[key]
    return float(min(spec.maximum, max(spec.minimum, value)))


def _guess_name(prompt: str) -> str:
    words = [
        w for w in re.findall(r"[A-Za-z][A-Za-z'-]*", prompt) if w.lower() not in _STOP
    ]
    picked = words[:3] if words else ["Designed", "Voice"]
    return " ".join(word.capitalize() for word in picked)


class LexiconDesigner:
    designer_id = "lexicon"

    def __init__(self, macros: dict[str, Macro] | None = None) -> None:
        self._macros = macros if macros is not None else load_macros()

    def available(self) -> tuple[bool, str]:
        return True, ""

    def design(self, prompt: str) -> VoiceDesign:
        tokens = _tokens(prompt)
        tags: list[str] = []
        matched: list[str] = []
        nudges: dict[str, float] = {}
        weight = 1.0
        for token in tokens:
            if token in _MORE:
                weight = 1.5
                continue
            if token in _LESS:
                weight = 0.5
                continue
            for tag, words in _TAG_WORDS.items():
                if token in words and tag in self._macros:
                    if tag not in tags:
                        tags.append(tag)
                    matched.append(token)
                    break
            for key, delta in _NUDGE_WORDS.get(token, ()):
                nudges[key] = nudges.get(key, 0.0) + delta * weight
                if token not in matched:
                    matched.append(token)
            weight = 1.0
        defaults = {spec.key: spec.default for spec in SCHEMA}
        params = apply_macros(defaults, tags, self._macros)
        for key, delta in nudges.items():
            params[key] = params.get(key, defaults[key]) + delta
        explicit = re.search(
            r"([+-]?\d+(?:\.\d+)?)\s*(?:st|semitones?)", prompt.lower()
        )
        if explicit:
            params["pitch_semitones"] = float(explicit.group(1))
            matched.append(explicit.group(0))
        params = {key: _clip(key, float(value)) for key, value in params.items()}
        notes: list[str] = []
        lowered = prompt.lower()
        if any(word in tokens for word in _ACCENT_WORDS):
            notes.append(ACCENT_NOTE)
        if any(phrase in lowered for phrase in _IDENTITY_WORDS):
            notes.append(IDENTITY_NOTE)
        if not matched:
            notes.append(
                "No sound words recognised — try words like deep, tiny, gravelly, "
                "ghostly, whisper, echoing, distant."
            )
        return VoiceDesign(
            name=_guess_name(prompt),
            tone_tags=tags,
            params=params,
            matched=matched,
            notes=notes,
            designer=self.designer_id,
        )


def designers(macros: dict[str, Macro] | None = None) -> list[VoiceDesigner]:
    """Best first. The neural seam is listed so the UI can say why it is off."""
    from votr.neural import NeuralDesigner

    return [NeuralDesigner(), LexiconDesigner(macros)]


def design_voice(prompt: str, macros: dict[str, Macro] | None = None) -> VoiceDesign:
    """Use the first available designer (the lexicon one always is)."""
    for designer in designers(macros):
        ok, _reason = designer.available()
        if ok:
            return designer.design(prompt)
    raise RuntimeError("no Voice designer is available")

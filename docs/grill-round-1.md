# Grill — Round 1 (phase 1 of `grill-with-docs`) — ANSWERED

Status: **closed 2026-09-20.** All seven questions answered by Michael. Five answers
matched the recommendation; two overrode it (Q5 Preview, Q7 licence). Decisions that
passed the ADR gates are in `docs/adr/0002`–`0006`; everything else is recorded in
`docs/project-context.md`. Vocabulary is in `docs/CONTEXT.md`.

Method: `grilling` + `domain-modeling`. Facts were looked up (see
`docs/discovery-local-voice.md`); only the decisions were Michael's.

---

❓ **Q1 - Where does Roleplay Mode happen?** In-person, online (via a virtual cable into
Discord/Foundry), or both?

➡️ Recommended: both, in-person first.

✅ **Answer: Online only** (Discord or other online means). Not in-person. **Override.**
→ ADR-0002. Removes room acoustics/feedback from scope; Virtual Cable is the one
Roleplay output path.

---

❓ **Q2 - What machine will this run on?** OS, CPU, NVIDIA GPU/VRAM, target OSes.

➡️ Recommended: Windows 10/11 first, CPU-only baseline, NVIDIA optional.

✅ **Answer: Agree.** Windows 10/11 first; CPU-only baseline; NVIDIA optional. Michael's
GPU specifics still unknown — matters only for the later Neural Engine epic.

---

❓ **Q3 - What is a "character voice" in the rough draft?** (a) modulation of your own
voice, (b) a different person via neural VC, (c) a now, b later.

➡️ Recommended: (c) behind an `Engine` interface.

✅ **Answer: Agree — (c).** → ADR-0004.

---

❓ **Q4 - What must "personality + tone hints" do?** (a) notes, (b) tags → slider
macros, (c) free text generates a voice (GPU tier only).

➡️ Recommended: (a) + (b) now, keep the free-text field for (c).

✅ **Answer: Agree — a + b.** Free-text field kept for later GPU voice design. Folded into
ADR-0004 consequences.

---

❓ **Q5 - How should Preview work?** (a) live headphone monitor, (b) record a phrase and
replay through Voices, (c) both.

➡️ Recommended: (c), with (a) as the tuning loop.

✅ **Answer: Override — delayed Preview.** Record the utterance, then play it back
through the Voice on the GM's own speakers while tuning. Live monitor is *not* the
primary mechanic; design around record-then-play / play-on-a-delay. → ADR-0005.

---

❓ **Q6 - Latency tolerance and auto-tune.**

➡️ Recommended: auto-tune on first run, default ~120 ms, manual override.

✅ **Answer: Agree.** With online-only (Q1), 120 ms is a quality target, not a hard
ceiling.

---

❓ **Q7 - Stack and licence.** Python + PySide6 + JSON + PyInstaller; MIT vs GPL.

➡️ Recommended: MIT, Signalsmith Stretch, no GPL in the live path.

✅ **Answer: Override on licence — GPL**, explicitly to open Rubber Band / Pedalboard for
the live path. Stack stays Python + PySide6 + JSON + PyInstaller. → ADR-0003. Stack
table in `docs/discovery-local-voice.md` updated: Rubber Band `LiveShifter` becomes the
primary live pitch/formant option, Signalsmith Stretch the fallback.

---

## Round 2?

Not needed. The frontier that opened after these answers consists of routine calls
that do not change the design tree; I made them and recorded them as defaults in
`docs/project-context.md` ("Assumed defaults"). Override any of them by editing that
list or telling the coordinator:

- Preview Take is captured push-to-talk (hold to record, release to render and play),
  with a fallback bundled sample phrase when no mic is present.
- Roleplay Mode has an *optional* monitor toggle (Character Voice to the GM's headphones),
  off by default.
- Voice files are one JSON per Voice in the OS user-data folder; import/export is "copy
  the file" and is not in the rough draft.
- The Discord wizard targets VB-CABLE on Windows first; other cables are detected by
  device-name pattern.
- The rough draft is Windows-only for the installer; the code stays cross-platform.

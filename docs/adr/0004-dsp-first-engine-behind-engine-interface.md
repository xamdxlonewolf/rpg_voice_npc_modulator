---
status: accepted
---

# DSP modulation first, behind an Engine interface that a Neural Engine can implement later

The rough draft ships one Engine: a DSP Engine that modulates the GM's own voice
(pitch, formants, effects) on CPU. A Neural Engine that makes the GM sound like a
different person (zero-shot voice conversion such as X-VC, NVIDIA GPU only, 300–500 ms)
is a later, optional Engine behind the same interface. Decided in grill round 1 (Q3),
matching the recommendation.

## Considered options

- Neural first: the "wow" result, but excludes every CPU-only machine, adds multi-GB
  downloads, and depends on five-month-old research code.
- DSP only, forever: simplest, but permanently caps the app at "same actor, different
  mask" and makes the free-text Tone Hints field a dead end.

## Consequences

- The Engine interface is a stable seam: stream processing on fixed blocks, offline
  rendering for Preview, capability flags (can-change-identity, needs-GPU), and a
  parameter schema each Engine declares. UI and Voice storage must not depend on which
  Engine is active.
- A Voice records which Engine it targets; a Voice built for an Engine that is not
  installed is shown but not selectable.
- Tone Hints (Q4) are notes plus Tone Tags → slider Macros in the DSP Engine; the
  free-text field is kept so a Neural Engine can consume it for voice design later.

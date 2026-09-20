---
status: accepted
---

# Roleplay Mode targets online play only, delivered through a Virtual Cable

Roleplay Mode sends the Character Voice to a Virtual Cable so Discord (or any other
online tool) uses it as its microphone. In-person table play through room speakers is
not a supported Roleplay Mode target. Decided by Michael in grill round 1 (Q1),
overriding the recommendation to design in-person first.

## Considered options

- In-person via room speakers: players would still hear the GM's real voice, the mic
  would pick up the speaker (feedback), and the doubled-voice effect forces a hard
  ~120 ms latency ceiling. Solvable only with headsets and careful acoustics the app
  cannot control.
- Both: doubles the routing UI and testing surface for a rough draft.

## Consequences

- Output routing has one primary path: app → Virtual Cable → Discord input. Speakers
  are used only for Preview and optional monitoring.
- The latency bar is what online conversation tolerates (a few hundred ms), so the
  ~120 ms default (Q6) is a quality target, not a hard requirement.
- Feedback suppression and room-acoustics features are out of scope.
- A "set up for Discord" wizard is a rough-draft requirement, not a nice-to-have.

---
status: accepted
---

# The app is licensed under the GPL

Voice of the Realm is released under the GNU GPL (v3, or v2-or-later where a dependency
requires compatibility). Michael chose GPL in grill round 1 (Q7), overriding the MIT
recommendation, specifically to allow GPL-only audio libraries in the live processing
path: Rubber Band (`LiveShifter`, GPLv2+) and Spotify Pedalboard (GPLv3, which also
bundles JUCE and Rubber Band).

## Considered options

- MIT: maximum reuse by others, but confines the live pitch/formant path to
  permissively licensed code (Signalsmith Stretch) and forbids Pedalboard entirely.
- GPL: unlocks the best-in-class live pitch shifter with formant preservation and a
  ready-made effects library, at the cost of requiring derivative works to stay GPL.

## Consequences

- Every bundled dependency must be GPL-compatible; the installer ships the GPL text and
  a third-party notices file (Rubber Band, FFTW, JUCE, PySide6/Qt LGPL, PortAudio, etc.).
- PySide6/Qt is used under the LGPL, which is compatible; no Qt commercial licence is
  needed as long as Qt is dynamically linked (PyInstaller `--onedir` does this).
- GPL virtual-cable drivers (BlackHole) could in principle be bundled, but see ADR-0006:
  we still link rather than bundle.
- Reversing this later would mean removing Rubber Band/Pedalboard from the live path.

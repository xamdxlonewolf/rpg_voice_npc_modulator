# Third-party notices

Voice of the Realm is licensed under the GNU GPL v3 (or later). This file lists
third-party software that the app depends on or will bundle. Notices will be
completed as each dependency is added to a release. Virtual Cable drivers are
**not** bundled (see `docs/adr/0006-link-to-not-bundle-virtual-cable-drivers.md`).

## Qt / PySide6

- Project: [Qt](https://www.qt.io/) / [PySide6](https://doc.qt.io/qtforpython/)
- License: GNU LGPL v3 (dynamically linked)
- Used for: desktop UI

## PortAudio (via sounddevice, planned)

- Project: [PortAudio](https://www.portaudio.com/)
- License: MIT-style
- Used for: microphone and device I/O (not wired in S0.1)

## Rubber Band Library (planned)

- Project: [Rubber Band](https://breakfastquay.com/rubberband/)
- License: GPLv2 or later
- Used for: live pitch/formant shifting (not wired in S0.1)

## FFTW (via Rubber Band, planned)

- Project: [FFTW](https://www.fftw.org/)
- License: GPLv2 or later
- Used for: FFT support inside Rubber Band

## Pedalboard (planned)

- Project: [spotify/pedalboard](https://github.com/spotify/pedalboard)
- License: GPLv3
- Used for: effects (reverb, filters, compressor)

## JUCE (via Pedalboard, planned)

- Project: [JUCE](https://juce.com/)
- License: GPLv3 (as used by Pedalboard)
- Used for: audio processing internals inside Pedalboard

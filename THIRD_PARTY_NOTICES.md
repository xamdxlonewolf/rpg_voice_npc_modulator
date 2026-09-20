# Third-party notices

Voice of the Realm is licensed under the GNU GPL v3 (or later). This file lists
third-party software that the app depends on or will bundle. Notices will be
completed as each dependency is added to a release. Virtual Cable drivers are
**not** bundled (see `docs/adr/0006-link-to-not-bundle-virtual-cable-drivers.md`).

## Qt / PySide6

- Project: [Qt](https://www.qt.io/) / [PySide6](https://doc.qt.io/qtforpython/)
- License: GNU LGPL v3 (dynamically linked)
- Used for: desktop UI

## PortAudio (via sounddevice)

- Project: [PortAudio](https://www.portaudio.com/) / [sounddevice](https://python-sounddevice.readthedocs.io/)
- License: MIT-style
- Used for: duplex microphone and device I/O

## Signalsmith Stretch (via python-stretch)

- Project: [Signalsmith Stretch](https://signalsmith-audio.co.uk/code/stretch/) / [python-stretch](https://pypi.org/project/python-stretch/)
- License: MIT
- Used for: fallback live pitch shift (Windows wheels available)

## Rubber Band Library (via rubband, optional)

- Project: [Rubber Band](https://breakfastquay.com/rubberband/) / [rubband](https://pypi.org/project/rubband/)
- License: GPLv2 or later
- Used for: intended live pitch/formant core (`LiveShifter`). No published Windows wheel as of 0.3.1.

## FFTW (via Rubber Band, planned)

- Project: [FFTW](https://www.fftw.org/)
- License: GPLv2 or later
- Used for: FFT support inside Rubber Band

## Pedalboard

- Project: [spotify/pedalboard](https://github.com/spotify/pedalboard)
- License: GPLv3
- Used for: block-wise live effects (reverb, low-pass, compressor) with `reset=False`

## JUCE (via Pedalboard, planned)

- Project: [JUCE](https://juce.com/)
- License: GPLv3 (as used by Pedalboard)
- Used for: audio processing internals inside Pedalboard

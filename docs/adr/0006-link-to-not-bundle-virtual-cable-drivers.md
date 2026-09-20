---
status: accepted
---

# Virtual Cable drivers are linked to and detected, never bundled or installed by the app

The app does not ship or silently install a Virtual Cable driver. It detects whether a
known cable (VB-CABLE on Windows, BlackHole on macOS, a PipeWire/PulseAudio null sink on
Linux) is present, and if not, a guided wizard links the GM to the vendor download and
verifies the device afterwards.

## Considered options

- Bundle VB-CABLE: it is donationware whose redistribution and volume use require a
  licence from VB-Audio; it also needs an admin-elevated driver install and a reboot,
  which would make our installer heavier and fragile.
- Bundle BlackHole (GPL, so licence-compatible with ADR-0003): macOS is not the first
  target and kernel-adjacent driver packaging is out of proportion for a rough draft.
- Write our own virtual audio device: a signed kernel driver on Windows; far beyond
  scope.

## Consequences

- Roleplay Mode has a hard external prerequisite that the app must explain clearly and
  check for on startup and before entering Roleplay Mode.
- The Discord setup wizard is part of the rough draft (see ADR-0002).
- Because Linux needs no download, the wizard's Linux path can create the null sink
  itself with `pactl` (no driver involved); this does not violate the decision.

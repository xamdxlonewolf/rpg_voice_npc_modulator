---
status: accepted
---

# Local-only and free: no cloud storage, no cloud or paid AI services

Voice of the Realm must be free to build and free to use, and a GM's Voices must never
leave their machine. All processing (DSP or neural) runs on the user's computer, all
Voices are stored on the user's disk, and the app depends only on free, openly licensed
libraries and model weights. This rules out ElevenLabs-class voice design, cloud
voice-conversion APIs, hosted TTS, and free tiers that meter or expire.

## Considered options

- Cloud voice design / conversion (ElevenLabs, Azure, Respeecher): far better "sounds
  like a different person" quality, near-zero setup, but paid per use, requires an
  account and internet, and sends the GM's voice off-device.
- Hybrid (local DSP, cloud AI on demand): rejected because any paid or metered path
  breaks the "free for users" promise and makes the app's behaviour depend on a vendor.

## Consequences

- Neural voice conversion is only available on machines with a capable GPU; CPU-only
  machines get DSP modulation of the GM's own voice.
- Model and library licences must be permissive or at least free for end users;
  non-commercial-only or unclear-licence models are excluded.
- No accounts, telemetry, or sync features.

# S6.4 — Clean-machine acceptance (rough draft)

**Status: not run.** This is a checklist, not a completed acceptance run.
Do not treat anything below as a measured result.

Target: a clean Windows 10/11 machine with VB-CABLE installed (user installs
the cable from the vendor; the app never bundles it).

## Checklist (from `docs/backlog.md` definition of done)

1. [ ] Install Voice of the Realm from the Inno Setup per-user installer
      (Start Menu shortcut; no `git clone`). Confirm the wizard said this is
      the DSP app and that Neural is a separate source/`[neural]` extra.
2. [ ] First-run setup completes (or is skipped and re-run from Settings).
3. [ ] Devices: mic and speakers picked; Virtual Cable detected.
4. [ ] Latency Test reports a method. If glass-to-glass cannot run, the UI
      must say so — no invented millisecond number.
5. [ ] Discord wizard: Input Device → CABLE Output; test tone visible in Discord.
6. [ ] Create a Voice named **Grimjaw** with Tone Tags `gravelly` and `booming`.
7. [ ] Save it; it appears as a Voice Button.
8. [ ] Record a Take; hear it as Grimjaw on the speakers (delayed Preview).
9. [ ] Click the Grimjaw Voice Button (Active Voice).
10. [ ] Enter Roleplay Mode.
11. [ ] A Player in Discord hears **only** Grimjaw (never the Dry Voice).
12. [ ] Panic mute (`Ctrl+Shift+M` in-app) silences the cable; second press unmutes.

## Recorded results

| Item | Result |
| --- | --- |
| Machine | *not run — needs a clean Windows 10/11 VM* |
| Installer version | *—* |
| Latency Test method | *—* |
| Glass-to-glass ms | *—* |
| Discord heard Character Voice only | *—* |

Re-run this list on a fresh Windows VM after S6.1–S6.3 produce an installer
on that OS. Linux cloud VMs cannot execute this acceptance.

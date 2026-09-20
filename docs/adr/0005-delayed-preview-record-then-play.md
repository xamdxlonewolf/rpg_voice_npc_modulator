---
status: accepted
---

# Preview is delayed: record a Take, then play it back through the Voice on the GM's speakers

Preview records the GM's utterance (a Take) and then plays it back through the selected
Voice on the GM's own speakers, so the GM hears the result while tuning. A live
headphone monitor is not the primary Preview mechanic. Decided by Michael in grill
round 1 (Q5), overriding the recommendation to lead with a live monitor.

## Considered options

- Live monitor in headphones: instant feedback while moving sliders, but requires
  headphones (speakers would feed back into the mic), and hearing your own voice with
  ~120 ms delay is disorienting for many people.
- Record-then-play: one extra step, but works on speakers, needs no headphones, lets
  the GM concentrate on listening rather than talking, and the same Take can be
  replayed through several Voices for A/B comparison.

## Consequences

- Preview playback goes to the GM's normal speakers/headphones, never to the Virtual
  Cable; Roleplay Mode and Preview therefore use different Output Devices.
- The Take must be rendered through the same streaming Engine path (block by block)
  used in Roleplay Mode, so what the GM hears in Preview is what players will hear.
  Offline "higher quality" rendering is deliberately not used for Preview.
- Slider changes re-render and replay the last Take (or a bundled sample phrase when no
  mic is present), so tuning stays a tight loop without a live monitor.
- A live monitor remains possible as an optional, secondary feature.

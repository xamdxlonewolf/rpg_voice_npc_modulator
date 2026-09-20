# Voice of the Realm

A local desktop app that lets a tabletop Game Master who plays online speak as their
characters through saved, previewable character voices.

## Language

### People

**GM**:
The person running the game and using the app; the only user.
_Avoid_: DM, host, streamer

**Player**:
Anyone hearing the GM through the online tool (Discord etc.). Not a user of the app.
_Avoid_: audience, listener, participant

### Voices

**Voice**:
A named, saved character sound the GM can speak through. Has Tone Hints, Tone Tags, and sound settings for one Engine.
_Avoid_: preset, profile, filter, skin, character (the character is the fiction; the Voice is its sound)

**Voice Library**:
The GM's collection of saved Voices, shown as Voice Buttons.
_Avoid_: soundboard, deck

**Voice Button**:
The clickable tile that selects one Voice. Clicking it makes that Voice the Active Voice.
_Avoid_: tile, card, pad

**Active Voice**:
The single Voice currently applied to the GM's speech.
_Avoid_: selected voice, current preset

**Tone Hints**:
Free-text personality and tone notes attached to a Voice, written by the GM for the GM.
_Avoid_: prompt, description, metadata

**Tone Tag**:
A structured word attached to a Voice (e.g. "gravelly", "booming", "frail") that applies a Macro.
_Avoid_: label, keyword, style

**Macro**:
A named bundle of sound-setting values that a Tone Tag applies to a Voice's sliders.
_Avoid_: preset, template, recipe

### Engines

**Engine**:
A processing method that turns the Dry Voice into a Character Voice. Each Voice targets one Engine.
_Avoid_: backend, model, pipeline

**DSP Engine**:
The Engine that modulates the GM's own voice (pitch, formants, effects) on any CPU. The only Engine in the rough draft.
_Avoid_: modulator, effects engine, voice changer

**Neural Engine**:
A later, optional Engine that makes the GM sound like a different person; needs a GPU.
_Avoid_: AI engine, voice cloner, converter

### Audio

**Dry Voice**:
The GM's own unprocessed microphone signal.
_Avoid_: raw input, original, source

**Character Voice**:
The processed signal after the Active Voice is applied to the Dry Voice.
_Avoid_: wet signal, output, converted audio

**Output Device**:
Where audio is sent: the Virtual Cable in Roleplay Mode, the GM's speakers in Preview.
_Avoid_: sink, playback device

**Virtual Cable**:
A software audio device that lets another app (e.g. Discord) use the Character Voice as its microphone.
_Avoid_: loopback, virtual mic, null sink

**Latency**:
The delay between the GM speaking and the Character Voice leaving the Output Device.
_Avoid_: lag, delay

**Latency Test**:
The first-run measurement that picks the app's default Latency settings.
_Avoid_: calibration, benchmark

### Modes

**Preview**:
Hearing a Voice by recording a Take and playing it back through that Voice on the GM's speakers.
_Avoid_: test, audition, live monitor

**Take**:
A short recording of the GM's speech used for Preview; can be replayed through any Voice.
_Avoid_: sample, clip, recording, utterance

**Roleplay Mode**:
The live state in which the Dry Voice is never sent to the Output Device and only the Character Voice reaches the Virtual Cable.
_Avoid_: live mode, performance mode, on air

**Monitor**:
Optionally hearing the Character Voice in the GM's own headphones during Roleplay Mode.
_Avoid_: sidetone, self-listen, playback

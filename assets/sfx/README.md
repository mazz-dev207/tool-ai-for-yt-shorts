# Optional V3 SFX

Place optional local sound-design assets in this directory.

Examples:

```text
impact.wav
soft_accent.wav
whoosh.wav
```

V3 never downloads or invents SFX automatically. Gemini may propose an `audio_event`, but the deterministic executor only uses an asset if the referenced local file exists.

Missing assets are logged and skipped. The original dialogue remains available and the pipeline continues.

Recommended source files:

- WAV or high-quality audio;
- short duration;
- no clipping;
- conservative loudness;
- content you have the right to use.

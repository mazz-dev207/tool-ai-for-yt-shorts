# CHANGELOG — AI Shorts V3

## Base reused from AI Shorts V2

The V3 branch starts from `upgrade/hook-optimizer-v2`. Existing working systems are reused rather than duplicated:

- `src/transcribe.py` — Whisper transcription and word timestamps.
- `src/chunk_transcript.py` — transcript analysis windows.
- `src/highlights/` — candidate generation, Gemini Multimodal Judge, ranking and profiles.
- `src/retention/` — retention optimization and timestamp remapping.
- `src/hooks/` V2 modules — Hook Start Optimizer and local boundary refinement.
- `src/smart_cut.py` — sentence/word/dead-air/payoff/reaction boundary analysis.
- `src/smart_crop.py`, `src/smart_crop_v2.py`, `src/smart_crop_profiles.py` — existing SmartCrop stack.
- `src/webcam_detector.py`, `src/forced_webcam_mode.py` — robust and forced webcam modes.
- `src/gameplay_tracker*.py`, `src/reading_focus.py` — stable gameplay and reading focus.
- `src/smartcrop_compositor.py` — gameplay/webcam vertical compositor with blurred fill.
- `src/caption_engine.py`, `src/caption/` — captions, grouping, ASS and karaoke.
- `src/renderer.py` — existing SmartCrop + subtitle FFmpeg renderer.

## Modified

### `main.py`

Added the V3 orchestration layer while retaining the ability to run V2 from the V3 checkout.

New CLI:

```text
--v3-mode auto|on|off
```

V3 pipeline insertion:

```text
Retention
→ Hook Optimizer V2
→ V3 Editorial Stage
→ SmartCut 3
→ Captions/Karaoke
→ SmartCrop/Render
→ Semantic Post-Processor
```

## Created — Configuration

### `src/v3_config.py`

Owns V3-only feature flags, thresholds, cache location and editorial profile path without rewriting the existing V2 config module.

### `config/editorial_profile.json`

Adds the `mazclips` editorial identity with content-specific edit-density rules.

## Created — Typed editorial model

### `src/v3/models.py`

Defines and validates:

- `HookPlan`
- `TimelineSegment`
- `VisualEvent`
- `AudioEvent`
- `ContextOverlay`
- `CropInstruction`
- `TransitionInstruction`
- `OriginalityResult`
- `OriginalityQAReport`
- `EditPlan`

Responsibilities include timestamp validity, known-enum validation, duplicate-content rules, effect budget and audio limits.

## Created — V3 multimodal reasoning

### `src/v3/editorial_reasoner.py`

Uses one structured Gemini multimodal request per selected Short to reason about:

- why the moment is interesting;
- editorial angle;
- Native/Reconstructed/Editorial hooks;
- story timeline;
- context overlays;
- semantic visual events;
- sound-design events;
- caption emphasis;
- transformation recommendations.

Includes structured JSON schema, cache, retry and quota handling.

### `src/v3/pipeline.py`

Coordinates the V3 stage and bounded regeneration loop. Writes debug artifacts and chooses the best valid plan when the originality threshold cannot be reached.

## Created — Originality Engine

### `src/originality/models.py`

Typed `EditorialAngle` and `OriginalityAnalysis` data.

### `src/originality/analyzer.py`

Normalizes source dependency, context independence, opening quality, transformation need and recommendations. Also provides the exact transcript truth context to deterministic overlay validators.

### `src/originality/angle_generator.py`

Normalizes Gemini's content-derived editorial angle.

### `src/originality/transformation_plan.py`

Maps free editorial recommendations to a finite known transformation vocabulary.

### `src/originality/scoring.py`

Calculates deterministic Originality Score from editorial/narrative/context/visual/hook/audio/brand/source-dependency components.

## Created — Hook V3

### `src/hooks/v3.py`

Extends, rather than replaces, Hook Optimizer V2.

Features:

- `NATIVE` — existing V2 Hook Optimizer start.
- `RECONSTRUCTED` — real source quote/reaction used as cold open.
- `EDITORIAL` — short truthful on-screen context hook.

Safety:

- reconstructed quotes must match transcript;
- source range must be inside analyzed context;
- numeric facts must exist in transcript;
- high-risk factual/relationship/outcome/intent terms must be transcript-supported;
- generic clickbait formulas are rejected;
- same guards can filter context overlays.

## Created — Story Restructurer

### `src/story/restructurer.py`

Builds explicit editorial source segments in the order the Short should play.

Supports purposes:

```text
cold_open
hook
context
setup
escalation
payoff
reaction
aftermath
callback
replay
```

Allows duplicate source ranges only for cold open/replay/callback and never sorts the final editorial order by source timestamp.

Playback-rate proposals are currently kept at 1.0x to preserve audio/caption/karaoke synchronization.

## Created — Edit planning/execution

### `src/editing/planner.py`

Connects HookPlan + EditorialAngle + Story Timeline + event instructions into one typed `EditPlan`.

A Reconstructed Hook can be prepended as a real source `cold_open` segment.

### `src/editing/smartcut_v3.py`

Executes editorial timeline order. Reuses the V2 SmartCut boundary engine on each source segment separately and then concatenates those segments in the planned order.

### `src/editing/visual_grammar.py`

Maps semantic events to a controlled visual vocabulary and enforces event-level/global effect budgets.

### `src/editing/sound_design.py`

Normalizes optional semantic sound events with safe duration/intensity/gain limits.

### `src/editing/post_processor.py`

Executes supported semantic edits after the stable V2 vertical render:

- truthful context overlays through ASS;
- center punch-ins through FFmpeg `sendcmd` + named crop filter;
- optional local SFX mixing with output limiter;
- fail-safe preservation of the existing rendered clip on error.

### `src/editing/originality_qa.py`

Checks plan validity, effect budget, hook quality, context independence, audio safety, payoff identification and originality threshold.

Strong standalone `NO TRANSFORMATION NEEDED` plans can intentionally pass without artificial effects.

## Created — Tests

### `tests/test_v3_models.py`

- valid EditPlan;
- effect-budget violation;
- duplicate-content validation.

### `tests/test_originality_scoring_v3.py`

- weighted score range;
- transformed editorial plan versus plain extraction.

### `tests/test_story_restructurer_v3.py`

- reordered timeline preservation;
- invalid source timeline fallback.

### `tests/test_hook_v3.py`

- real reconstructed quote selection;
- invented numeric editorial stake rejection.

### `tests/test_sound_design_v3.py`

- sound effect gain cap.

### `tests/test_v3_post_processor.py`

- semantic punch-in creates FFmpeg crop runtime commands;
- unsupported visual event is not executed arbitrarily.

### `tests/test_originality_qa_v3.py`

- low originality requests regeneration;
- strong standalone no-transformation plan can pass intentionally.

## Documentation

### `README_AI_SHORTS_V3.md`

Full architecture, run instructions, configuration, debug files, fallback design and known limitations.

### `CHANGELOG_V3.md`

This file: detailed created/modified/reused module responsibilities.

## Backward compatibility

The V3 checkout can disable editorial features with:

```text
--v3-mode off
```

This keeps the inherited V2 flow available for comparison and fallback testing.

## Known first-release limitations

- Semantic effects are currently a post-render pass, preserving renderer stability but also scaling already-burned captions during punch-ins.
- Freeze-frame is representable in `EditPlan` but not yet executed by the deterministic post-processor.
- Playback-rate/speed-ramp execution is disabled until timestamps/audio can be safely retimed.
- SFX execution requires optional local assets in `assets/sfx/`.
- Subjective editorial quality still needs real-video evaluation in addition to deterministic unit tests.

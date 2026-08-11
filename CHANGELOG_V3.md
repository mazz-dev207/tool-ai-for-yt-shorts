# CHANGELOG — AI Shorts V3

## Base reused from AI Shorts V2

The V3 branch starts from the working Hook Optimizer V2 baseline. Existing working systems are reused rather than duplicated:

- `src/transcribe.py` — Whisper transcription and word timestamps.
- `src/chunk_transcript.py` — transcript analysis windows.
- `src/highlights/` — candidate generation, Gemini Multimodal Judge, ranking and profiles.
- `src/retention/` — retention optimization and timestamp remapping.
- `src/hooks/` V2 modules — Hook Start Optimizer and local boundary refinement.
- `src/smart_cut.py` — sentence/word/dead-air/payoff/reaction boundary analysis.
- `src/smart_crop.py`, `src/smart_crop_v2.py`, `src/smart_crop_profiles.py` — SmartCrop stack.
- `src/webcam_detector.py`, `src/forced_webcam_mode.py` — robust and forced webcam modes.
- `src/gameplay_tracker*.py`, `src/reading_focus.py` — stable gameplay and Reading Focus.
- `src/smartcrop_compositor.py` — gameplay/webcam vertical compositor with blurred fill.
- `src/caption_engine.py`, `src/caption/` — captions, grouping, ASS and karaoke.
- `src/renderer.py` — existing SmartCrop + subtitle FFmpeg renderer.

# Modified

## `main.py`

V3 orchestration now includes the strategic layer before expensive Gemini/highlight processing.

Current high-level flow:

```text
Candidate Discovery
→ Format Intelligence / Strategic Gate
→ Gemini Highlight Judge
→ Retention
→ Hook Optimizer V2
→ V3 Editorial Stage
→ SmartCut 3
→ Captions/Karaoke
→ SmartCrop/Render
→ Semantic Post-Processor
→ Experiment Metadata
```

Added CLI:

```text
--v3-mode auto|on|off
--channel-strategy PATH
--format-brief PATH
--format-id FORMAT_ID
--skip-low-format-fit
```

`--v3-mode off` skips strategic/editorial V3 work and keeps the inherited V2 path available.

## `src/highlights/gemini_judge.py`

The existing Gemini Highlight Judge is now format-aware when strategy metadata is present.

Changes:

- receives `ChannelStrategy`, `FormatProfile`, format match and opportunity metadata through the candidate;
- asks whether required format signals are actually visible/audible;
- considers channel suitability separately from generic interestingness;
- checks whether payoff justifies setup and whether the Short can become self-contained;
- explicitly forbids invented stakes/events/demand claims;
- treats Format Fit and Content Opportunity as evidence rather than mechanically adding them to Highlight Score;
- includes strategy context in the Gemini cache key so changing strategy invalidates stale judgements.

# Created — Strategic configuration

## `config/channel_strategy_mazclips_gaming.json`

Defines MazClips Gaming publishing strategy:

- primary niche and positioning;
- priority subniches;
- priority repeatable formats;
- patterns to avoid;
- minimum Format Fit and Content Opportunity thresholds.

Initial thresholds:

```text
min_format_fit = 45
min_content_opportunity = 50
```

## `config/formats/mazclips_gaming_formats.json`

Initial configurable seed library:

```text
gaming_high_stakes_fail_reaction
gaming_clutch_comeback
gaming_visual_surprise
gaming_funny_player_interaction
gaming_challenge_attempt
gaming_rage_emotional_reversal
```

Each format includes required signals, preferred hooks/story shape, avoid rules, target duration, version and enabled status.

## `config/format_brief.example.json`

Documents the stable JSON contract for optional manual/Trend Finder input without coupling the two codebases.

# Created — Channel Strategy / Format Intelligence

## `src/strategy/models.py`

Typed dataclasses:

- `ChannelStrategy`
- `FormatProfile`
- `FormatBrief`
- `FormatMatch`
- `ContentOpportunityScore`
- `ExperimentMetadata`

The models validate required fields and clamp numeric scores.

## `src/strategy/library.py`

Responsibilities:

- loads format libraries from JSON;
- loads ChannelStrategy;
- loads imported FormatBrief;
- resolves default MazClips Gaming strategy;
- provides safe fallback strategy for other profiles.

No direct dependency on Trend Finder exists.

## `src/strategy/format_intelligence.py`

Implements the deterministic strategic pre-filter layer.

Responsibilities:

- infer evidence/signals from candidate transcript and existing discovery metadata;
- compare a candidate with enabled FormatProfiles;
- support manual format, imported brief and auto-classify modes;
- calculate `format_fit_score`;
- calculate separate `content_opportunity_score`;
- attach `strategy_context` to candidate metadata;
- mark `PASS` or `LOW FORMAT FIT`;
- optionally skip weak candidates;
- preserve all candidates if strict filtering would otherwise produce zero output;
- write `highlights/<video>_format_metadata.json`.

Content Opportunity components:

```text
Format Match                 25%
Hook Potential               15%
Payoff Strength              15%
Context Independence         10%
Emotional / Visual Signal    10%
Source Quality               10%
Transformability             10%
Channel Fit                   5%
```

Imported external demand remains separate and is not treated as predicted views.

## `src/strategy/experiments.py`

Exports format/experiment metadata after rendering.

Per Short:

```text
format_profile.json
format_match.json
content_opportunity.json
channel_strategy.json
experiment_metadata.json
```

Aggregate:

```text
output/debug/<video>/experiment_metadata.json
```

Performance fields are intentionally initialized as null for later analytics import.

# Created — V3 configuration

## `src/v3_config.py`

Owns V3-only feature flags, thresholds, cache location and editorial profile path without replacing V2 configuration.

## `config/editorial_profile.json`

Defines MazClips editorial identity and content-specific edit-density rules.

ChannelStrategy answers **what stories are published**; Editorial Profile answers **how those stories look and feel**.

# Created — Typed editorial model

## `src/v3/models.py`

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

# Created — V3 multimodal reasoning

## `src/v3/editorial_reasoner.py`

Uses structured Gemini multimodal reasoning for final selected Shorts:

- why the moment is interesting;
- editorial angle;
- Native/Reconstructed/Editorial hooks;
- story timeline;
- context overlays;
- semantic visual events;
- sound-design events;
- caption emphasis;
- transformation recommendations.

Includes JSON schema, cache, retry and quota handling.

## `src/v3/pipeline.py`

Coordinates:

```text
Editorial Reasoning
→ Angle
→ Originality Analysis
→ Hook V3
→ Story Restructurer
→ EditPlan
→ Originality Score
→ QA / bounded regeneration
```

Writes debug artifacts and falls back to a V2-compatible plan if required.

# Created — Originality Engine

## `src/originality/models.py`

Typed editorial/originality data structures.

## `src/originality/analyzer.py`

Normalizes source dependency, context independence, opening quality and transformation need.

## `src/originality/angle_generator.py`

Normalizes Gemini's content-derived editorial angle.

## `src/originality/transformation_plan.py`

Maps recommendations to a finite known transformation vocabulary.

## `src/originality/scoring.py`

Calculates deterministic Originality Score from editorial, narrative, context, visual, hook, audio, brand and source-dependency components.

# Created — Hook V3

## `src/hooks/v3.py`

Extends rather than replaces Hook Optimizer V2.

Modes:

```text
NATIVE
RECONSTRUCTED
EDITORIAL
```

Safety guards require reconstructed quotes to exist in transcript and reject unsupported high-risk factual editorial hooks.

# Created — Story Restructurer

## `src/story/restructurer.py`

Builds explicit source segments in editorial playback order.

Supported purposes include:

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

Duplicate source ranges are permitted only for explicit editorial use cases such as cold open/replay/callback.

# Created — Edit planning/execution

## `src/editing/planner.py`

Builds one typed `EditPlan` from HookPlan, EditorialAngle, timeline and validated semantic event instructions.

## `src/editing/smartcut_v3.py`

Executes editorial timeline order while reusing V2 boundary analysis on each source segment.

## `src/editing/visual_grammar.py`

Maps semantic events to a controlled vocabulary and enforces effect budgets.

## `src/editing/sound_design.py`

Normalizes optional semantic sound events with conservative duration/intensity/gain limits.

## `src/editing/post_processor.py`

Executes supported semantic post-render operations:

- context overlays;
- semantic punch-ins;
- optional safe SFX;
- safe preservation of the existing render on failure.

## `src/editing/originality_qa.py`

Checks plan validity, effect budget, hook quality, context independence, audio safety, payoff identification and originality threshold.

# Created — Tests

Existing V3 deterministic suites cover:

- typed EditPlan validation;
- Originality Score;
- Story Restructurer;
- Hook V3 truth rules;
- sound-design limits;
- post-processor effect execution;
- Originality QA;
- caption remapping;
- execution/fallback gates;
- content truth and source bounds.

## `tests/test_strategy_v3.py`

Adds strategic tests for:

1. required-signal format matching;
2. Content Opportunity as a separate score;
3. manual FormatProfile selection;
4. external demand remaining separate from the opportunity score;
5. candidate strategic annotation;
6. format-aware Gemini prompt and truth safeguards;
7. per-Short strategy/experiment artifact export.

Tests use mocks/temp files and do not require real Gemini API calls.

# Documentation

## `README_AI_SHORTS_V3.md`

Documents:

- separate V3 checkout;
- full format-aware pipeline;
- ChannelStrategy vs Editorial Brand;
- FormatProfile library;
- manual/imported/auto classification modes;
- Content Opportunity Score;
- strategic gate;
- Originality/Hook/Story/EditPlan architecture;
- debug and experiment metadata;
- PowerShell run/test commands;
- fallback behavior and limitations.

## `CHANGELOG_V3.md`

This file lists created, modified and reused modules with their responsibilities.

# Backward compatibility

The V3 checkout can disable all V3 strategic/editorial behavior with:

```text
--v3-mode off
```

The inherited V2 flow remains available for comparison and fallback testing.

# Known first-release limitations

- Format matching begins with deterministic transcript/metadata evidence; Gemini later performs multimodal verification.
- Initial configured format library is gaming-focused and intentionally small.
- Content Opportunity Score is not a predicted-view or guaranteed-virality metric.
- Semantic effects are currently a post-render pass, preserving renderer stability but potentially scaling already-burned captions during punch-ins.
- Freeze-frame is representable in `EditPlan` but not yet automatically executed by the deterministic post-processor.
- Playback-rate/speed-ramp execution remains disabled until timestamps/audio can be safely retimed.
- Optional SFX require local assets in `assets/sfx/`.
- Subjective editorial quality still requires real-video evaluation after deterministic tests pass.

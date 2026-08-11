# AI Shorts V3

`ai-shorts-v3` transforms the existing local Python AI Shorts pipeline from a highlight extractor into a **format-aware AI short-form editor**.

The original V2 pipeline remains the safe fallback path. V3 adds two distinct layers:

```text
CONTENT STRATEGY
→ What kind of repeatable Short should this channel make?

EDITORIAL EXECUTION
→ How should this specific Short be constructed?
```

The system is designed to find the right kind of moment for the channel, construct a self-contained story around it, execute deterministic edits, validate originality and export experiment metadata for later performance comparison.

## Separation from `ai-shorts`

Development lives on the dedicated Git branch:

```text
ai-shorts-v3
```

Clone this branch into a separate directory such as `F:\ai-shorts-v3`.

Do **not** merge it into the production `F:\ai-shorts` checkout merely to test V3.

## Existing architecture reused

V3 reuses working V2 components instead of rewriting them:

- Whisper transcription and word-level timestamps;
- Candidate Generator;
- Gemini Multimodal Highlight Judge;
- Highlight Ranking;
- Retention Optimizer;
- Hook Optimizer V2;
- SmartCut boundary analysis;
- SmartCrop 2.0;
- robust webcam detection;
- gameplay + webcam vertical compositor;
- stable gameplay virtual cameraman;
- Reading Focus;
- Caption Engine;
- speech-rhythm grouping;
- karaoke word highlighting;
- FFmpeg/NVENC rendering;
- JSON/debug outputs, caching and logging.

## V3 pipeline

```text
CHANNEL STRATEGY / FORMAT PROFILE
    ↓
Optional imported Trend / Reference brief
    ↓
FORMAT INTELLIGENCE ENGINE
    ├─ manual format profile
    ├─ imported FormatBrief
    └─ auto classify
    ↓
Candidate Discovery
    ↓
Format Match + Content Opportunity Score
    ↓
Strategic Gate
    ├─ PASS
    ├─ LOW FORMAT FIT
    └─ optional SKIP
    ↓
Gemini Multimodal Highlight Judge
    ↓
Highlight Ranking
    ↓
Retention Optimizer
    ↓
Hook Optimizer V2
    ↓
V3 Editorial Reasoner
    ├─ why is this interesting?
    ├─ editorial angle
    ├─ Hook V3 candidates
    ├─ story proposal
    └─ semantic edit events
    ↓
Truth + schema validation
    ↓
Story Restructurer
    ↓
Typed EditPlan
    ↓
Originality Score + QA
    ├─ pass
    └─ bounded regeneration with weaknesses
    ↓
SmartCut 3.0
    ↓
Subtitles + Karaoke
    ↓
Existing SmartCrop / Webcam / Tracking
    ↓
Existing FFmpeg Render
    ↓
V3 Semantic Post-Processor
    ├─ context overlays
    ├─ semantic punch-ins
    └─ optional safe SFX
    ↓
Experiment / Format Metadata
    ↓
FINAL VERTICAL SHORT
```

If a V3 enhancement fails, the system keeps a V2-compatible plan and continues.

# Channel Strategy and Format Intelligence

## ChannelStrategy

Channel strategy describes **what kinds of stories the channel publishes**. It is separate from the editorial brand, which describes how those stories look and feel.

Default gaming strategy:

```text
config/channel_strategy_mazclips_gaming.json
```

Current MazClips Gaming strategy:

```text
channel: MazClips Gaming
primary niche: gaming
positioning: fast, self-contained gaming moments with clear stakes,
             authentic reactions and a strong payoff

priority subniches:
- streamer_moments
- competitive_gameplay
- gaming_reactions

priority formats:
- gaming_high_stakes_fail_reaction
- gaming_clutch_comeback
- gaming_visual_surprise

min_format_fit: 45
min_content_opportunity: 50
```

The strategy is JSON-configurable and can be replaced without changing Python logic.

## FormatProfile

A `FormatProfile` describes a **repeatable content mechanic**, not merely a broad niche.

Example distinction:

```text
NICHE
Gaming

SUBNICHE
Streamer moments

FORMAT
High-Stakes Fail + Reaction
```

Format profiles are typed dataclasses defined in:

```text
src/strategy/models.py
```

The default library is:

```text
config/formats/mazclips_gaming_formats.json
```

Initial gaming formats:

```text
gaming_high_stakes_fail_reaction
gaming_clutch_comeback
gaming_visual_surprise
gaming_funny_player_interaction
gaming_challenge_attempt
gaming_rage_emotional_reversal
```

Each format can define:

- niche and subniche;
- required signals;
- preferred hook modes;
- preferred story shape;
- patterns to avoid;
- target duration;
- version;
- enabled/disabled status.

This allows formats to be added, removed, disabled or versioned without rewriting the main scoring engine.

## Three Format Intelligence modes

### MODE A — Manual FormatProfile

Force a configured format:

```powershell
python main.py video1 --content-profile gaming --v3-mode on --format-id gaming_high_stakes_fail_reaction
```

Only the selected format is evaluated.

### MODE B — Imported FormatBrief

V3 can consume a stable JSON contract exported by Trend Finder or authored manually:

```powershell
python main.py video1 --content-profile gaming --v3-mode on --format-brief .\input\format_brief.json
```

Example schema:

```json
{
  "channel": "MazClips Gaming",
  "target_niche": "gaming",
  "target_subniche": "streamer moments",
  "priority_formats": [
    "gaming_high_stakes_fail_reaction",
    "gaming_clutch_comeback",
    "gaming_visual_surprise"
  ],
  "reference_patterns": [],
  "source": "manual_or_trend_finder"
}
```

Trend Finder is therefore optional. V3 does not import or call Trend Finder code directly.

If a future brief contains an `external_demand_signal`, V3 stores it separately and never pretends it is a guaranteed view prediction.

### MODE C — Auto classify

When neither `--format-id` nor `--format-brief` is supplied, V3 compares each candidate against the configured format library and selects the best match.

```powershell
python main.py video1 --content-profile gaming --v3-mode on
```

## Format Match

Format matching currently combines deterministic evidence from:

- candidate title/hook/payoff/reason;
- transcript around the candidate;
- existing discovery scores;
- required format signals;
- target duration;
- channel priorities.

The result contains:

```text
format_id
format_version
format_fit_score
present_signals
missing_signals
recommendation
reason
```

Recommendations are:

```text
use
low_priority
skip
```

The deterministic matcher is intentionally conservative. Gemini later verifies the moment multimodally using video + audio + transcript.

## Format-aware Gemini Highlight Judge

When strategy metadata exists, the existing Gemini Highlight Judge now receives it explicitly.

Gemini is asked to consider:

- whether the intended repeatable format actually exists in the media;
- whether required signals are visible/audible;
- whether the payoff justifies the setup;
- whether the moment can become self-contained;
- whether the candidate fits the channel rather than merely being interesting in isolation.

Gemini is explicitly told not to invent missing stakes, events or demand signals and not to copy a reference creator's assets or identity.

The Gemini cache key includes `format_id`, format version and strategy context, so changing strategy invalidates stale strategic judgements.

# Content Opportunity Score

`Content Opportunity Score` is deliberately separate from:

```text
Highlight Score
Hook Score
Originality Score
```

It answers:

```text
Is this source/moment strategically worth turning into a Short for this channel?
```

Current deterministic components:

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

Important:

```text
Content Opportunity Score != predicted views
```

Imported external demand is stored separately rather than added silently to this score.

# Strategic Gate

Before expensive editorial processing, every candidate receives:

```text
PASS
```

or:

```text
LOW FORMAT FIT
```

For the default MazClips Gaming strategy the initial thresholds are:

```text
format_fit >= 45
content_opportunity >= 50
```

By default low-fit candidates are **annotated but preserved** so early tuning does not accidentally remove useful clips.

For strict filtering use:

```powershell
python main.py video1 --content-profile gaming --v3-mode on --skip-low-format-fit
```

Even in strict mode, if every candidate would be removed the safe fallback preserves the original candidate list. V3 does not force a zero-output pipeline merely because the strategic classifier is uncertain.

# V3 Editorial Models

`src/v3/models.py` contains typed dataclasses used between AI reasoning and deterministic execution:

- `EditPlan`
- `HookPlan`
- `TimelineSegment`
- `VisualEvent`
- `AudioEvent`
- `ContextOverlay`
- `CropInstruction`
- `TransitionInstruction`
- `OriginalityResult`
- `OriginalityQAReport`

Strategy models live in `src/strategy/models.py`:

- `ChannelStrategy`
- `FormatProfile`
- `FormatBrief`
- `FormatMatch`
- `ContentOpportunityScore`
- `ExperimentMetadata`

Gemini never generates arbitrary FFmpeg commands. It produces structured editorial intent; Python validates the intent and deterministic execution translates supported instructions into edits.

# Originality Engine

Files:

```text
src/originality/
├── analyzer.py
├── angle_generator.py
├── transformation_plan.py
├── scoring.py
└── models.py
```

The engine answers:

- why the selected moment is interesting;
- what a viewer needs to understand it;
- how dependent it is on the source video;
- whether the existing opening is already strong;
- which transformations would add editorial value.

`NO TRANSFORMATION NEEDED` is a valid outcome. A strong standalone source moment is not forced to use extra effects simply to inflate an originality score.

# Editorial Angle

The angle is content-derived rather than a fixed category. Examples include close call, unexpected failure, conflict, reveal, reaction, mistake, escalation or payoff.

Structured angle data can contain:

```text
primary_angle
secondary_angle
viewer_question
stakes
payoff
confidence
```

This feeds Hook V3, Story Restructurer and Edit Planner.

# Hook V3

Hook Optimizer V2 remains responsible for finding a strong native START. V3 compares additional opening strategies.

## NATIVE

Uses the existing Hook Optimizer result.

## RECONSTRUCTED

Uses a real source quote/reaction from later in the analyzed context as a cold open.

Safety rules:

- the quote must exist in the source transcript;
- `source_start/source_end` identifies the real source audio;
- it must remain inside the context Gemini actually analyzed;
- V3 never synthesizes a fake quote.

## EDITORIAL

Uses concise on-screen framing text supported by the source.

Generic formulas such as:

```text
YOU WON'T BELIEVE THIS
WAIT UNTIL THE END
CRAZY MOMENT
```

are rejected unless context genuinely supports equivalent framing.

A strong native hook is retained when transformed alternatives are not clearly better.

# Content Truth

Truth is a hard constraint.

Deterministic guards reject:

- reconstructed quotes not found in transcript;
- source ranges outside analyzed context;
- invented numeric claims;
- selected high-risk factual terms unsupported by transcript;
- duplicate source ranges unless explicitly used as cold open/replay/callback.

Gemini is instructed to use neutral wording when uncertain.

# Story Restructurer

File:

```text
src/story/restructurer.py
```

Target grammar:

```text
Cold Open
↓
Context
↓
Escalation
↓
Payoff
↓
Reaction / Aftermath
```

The final timeline can remain non-chronological when editorially valid. It is not sorted back into source order.

The existing caption engine already remaps word timestamps using the explicit segment order, so reconstructed timelines remain compatible with captions and karaoke.

Playback rate is currently forced to `1.0x` to protect source audio and word timestamp synchronization until a complete retiming layer exists.

# Edit Planner

File:

```text
src/editing/planner.py
```

Principle:

```text
AI decides WHAT should happen.
Python / FFmpeg decides HOW it happens.
```

The plan combines:

- HookPlan;
- editorial angle;
- validated source timeline;
- context overlays;
- semantic visual events;
- audio events;
- caption emphasis metadata;
- content/editorial metadata.

# SmartCut 3.0

File:

```text
src/editing/smartcut_v3.py
```

SmartCut 3.0 reuses existing SmartCut boundary logic on editorial source segments and concatenates the validated ranges in editorial order.

It preserves:

- word/sentence boundary refinement;
- dead-air cleanup;
- payoff/reaction awareness;
- FFmpeg/NVENC export;
- fallback behavior.

# Semantic Visual Editing

Files:

```text
src/editing/visual_grammar.py
src/editing/post_processor.py
```

Effects are semantic/event-driven, never timer-driven.

Examples:

```text
surprise / payoff  → punch_in
reaction           → face_zoom style punch-in
important detail   → focus_crop style punch-in
```

Unsupported effects are ignored safely rather than converted into arbitrary filters.

## Effect budget

`visual_grammar.py` limits effects based on semantic intensity and total plan density. Dialogue/podcast content receives a smaller visual budget.

Target:

```text
intentional > hyperactive
```

# Context Overlays

Context overlays are a separate ASS layer generated by the V3 post-processor.

Their goals are:

- clarify context;
- establish real stakes;
- reduce source dependency;
- make the Short understandable without the long-form source.

Overlay text passes through truth safeguards.

# Sound Design

Files:

```text
src/editing/sound_design.py
src/editing/post_processor.py
```

Sound design is optional. Audio events have conservative duration, intensity and gain bounds.

Optional assets:

```text
assets/sfx/
```

If an asset is missing, V3 logs a warning and continues.

# Existing SmartCrop remains authoritative

V3 preserves:

- face/person tracking;
- gameplay target tracking;
- Reading Focus;
- robust webcam detection;
- `GAMEPLAY_WEBCAM_STACK`;
- forced `--smartcrop-mode gameplay-webcam`;
- blurred panel fill;
- dead zones, smoothing, hysteresis and anti-ping-pong behavior.

# Originality Score

Originality is not a platform-detection bypass score. It measures meaningful editorial transformation and source independence.

Weights:

```text
Editorial Transformation     25%
Narrative Restructuring      20%
Context Independence         15%
Visual Transformation        15%
Hook Originality             10%
Audio Transformation          5%
Brand Consistency             5%
Inverse Source Dependency     5%
```

Interpretation:

```text
< 60   → QA requests regeneration
60–75  → acceptable
75–85  → strong
85+    → highly transformed
```

A strong `NO TRANSFORMATION NEEDED` plan can pass below 60 when the native content is already self-contained and editorially sound.

# Originality QA

File:

```text
src/editing/originality_qa.py
```

Checks include:

- timeline validity;
- hook quality;
- context independence;
- effect budget;
- audio safety;
- payoff preservation;
- originality threshold.

Retries are bounded by `MAX_EDIT_PLAN_RETRIES`. If the target remains unmet, the best valid plan is used with a warning.

# Editorial Brand Profile

Default:

```text
config/editorial_profile.json
```

The MazClips profile controls:

- hook style;
- caption style;
- semantic zoom behavior;
- sound design intensity;
- transition behavior;
- context overlays;
- ending behavior;
- editing density.

`ChannelStrategy` decides **what stories to publish**. `Editorial Brand` decides **how those stories feel visually/editorially**.

# Configuration

Main V3 flags:

```text
V3_ENABLED=true
ENABLE_ORIGINALITY_ENGINE=true
ENABLE_STORY_RESTRUCTURING=true
ENABLE_EDITORIAL_HOOKS=true
ENABLE_RECONSTRUCTED_HOOKS=true
ENABLE_CONTEXT_OVERLAYS=true
ENABLE_SEMANTIC_EFFECTS=true
ENABLE_SOUND_DESIGN=true
ENABLE_ORIGINALITY_QA=true

ORIGINALITY_MIN_SCORE=60
MAX_EDIT_PLAN_RETRIES=2
MAX_EFFECTS_PER_EVENT=2
V3_CONTEXT_BEFORE=8.0
V3_CONTEXT_AFTER=8.0
V3_EDITORIAL_PROFILE=mazclips
```

V2 flags remain valid.

# Debug artifacts

Per Short:

```text
output/debug/<video>/clip_<N>/
├── highlight.json
├── format_profile.json
├── format_match.json
├── content_opportunity.json
├── channel_strategy.json
├── hook_analysis.json
├── editorial_angle.json
├── originality_analysis.json
├── edit_plan.json
├── originality_qa.json
└── experiment_metadata.json
```

Aggregate strategy metadata:

```text
highlights/<video>_format_metadata.json
```

Aggregate experiment metadata:

```text
output/debug/<video>/experiment_metadata.json
```

Authoritative execution plan:

```text
highlights/v3/<video>/clip_<N>/edit_plan.json
```

# Experiment metadata / learning loop

Every final Short exports at least:

```text
short_id
format_id
format_version
format_fit_score
content_opportunity_score
strategic_gate
hook_type
hook_score
originality_score
duration
```

Performance fields are created but left `null` until real analytics are imported:

```text
views
viewed_vs_swiped
average_view_duration
average_percentage_viewed
likes
comments
shares
subscribers_gained
```

V3 does not automatically change channel strategy after one or two clips. The metadata exists so format performance can be compared over a meaningful sample later.

# Create the separate local project

From PowerShell:

```powershell
cd F:\
git clone --branch ai-shorts-v3 --single-branch https://github.com/mazz-dev207/tool-ai-for-yt-shorts.git F:\ai-shorts-v3
cd F:\ai-shorts-v3
```

Use a dedicated virtual environment. Copy only intentional local configuration such as `.env`; do not copy caches/output from V2.

Example:

```powershell
py -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item F:\ai-shorts\.env F:\ai-shorts-v3\.env
```

# How to run

## Auto-classify against MazClips Gaming formats

```powershell
python main.py video1 --highlight-mode legacy --content-profile gaming --smartcrop-mode gameplay-webcam --v3-mode on
```

## Force one format

```powershell
python main.py video1 --highlight-mode legacy --content-profile gaming --smartcrop-mode gameplay-webcam --v3-mode on --format-id gaming_high_stakes_fail_reaction
```

## Consume a FormatBrief exported by Trend Finder

```powershell
python main.py video1 --highlight-mode legacy --content-profile gaming --smartcrop-mode gameplay-webcam --v3-mode on --format-brief .\input\format_brief.json
```

## Use an explicit ChannelStrategy

```powershell
python main.py video1 --content-profile gaming --v3-mode on --channel-strategy .\config\channel_strategy_mazclips_gaming.json
```

## Enable strict strategic filtering

```powershell
python main.py video1 --content-profile gaming --v3-mode on --skip-low-format-fit
```

## Force V2 fallback path from V3 checkout

```powershell
python main.py video1 --highlight-mode legacy --content-profile gaming --smartcrop-mode gameplay-webcam --v3-mode off
```

# Tests

Strategy-specific deterministic tests:

```powershell
python -m unittest discover -s tests -p "test_strategy_v3.py" -v
```

All V3 tests:

```powershell
python -m unittest discover -s tests -p "test_*v3*.py" -v
```

Full inherited + V3 suite:

```powershell
python -m unittest discover -s tests -v
```

The deterministic V3 tests do not require real Gemini API calls.

# Fallback behavior

```text
Format Intelligence failure
→ keep candidate safely when possible

Gemini Highlight/Editorial failure
→ preserve V2-compatible highlight/hook/timeline

Story / EditPlan / Originality failure
→ use best valid or V2-compatible plan

Semantic post-processing failure
→ keep already-rendered vertical output
```

An enhancement should never destroy the whole pipeline.

# Known limitations

1. Format matching begins with deterministic transcript/metadata signals; Gemini supplies the later multimodal verification. It is not a perfect semantic classifier.
2. The initial format library is intentionally small and gaming-focused. Other channel strategies can provide their own JSON libraries/strategies as they are added.
3. `Content Opportunity Score` is a prioritization score, not a prediction of views or virality.
4. Semantic punch-ins and context overlays currently run as a safe post-render pass; a punch-in can also scale already-burned subtitles.
5. `freeze_frame` can be represented/validated but is not automatically executed by the initial post-processor. Story-level replay is supported through validated repeated source segments.
6. Playback-rate/speed-ramp execution is intentionally disabled until audio and word timestamps can be retimed together.
7. Content-truth validation is conservative but cannot guarantee full semantic entailment of every generated sentence.
8. Optional SFX require local audio assets under `assets/sfx/`.
9. Final visual/editorial quality still requires testing on real videos; unit tests validate deterministic behavior, not subjective taste.

# Fundamental principle

Do not copy another creator's assets. Learn repeatable mechanics and apply them to different source material while preserving MazClips identity.

```text
understand the channel strategy
→ identify the target repeatable format
→ evaluate whether the source contains that format
→ find the strongest matching moment
→ understand why viewers should care
→ build the hook
→ restructure the story truthfully
→ generate a typed EditPlan
→ execute deterministic semantic edits
→ validate originality
→ export format/experiment metadata
→ render a high-quality Short
```

> Don't just find a good moment. Find the right kind of moment for the channel, then turn it into a distinct piece of short-form content.

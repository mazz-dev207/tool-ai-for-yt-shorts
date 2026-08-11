# AI Shorts V3

`ai-shorts-v3` extends the existing local Python AI Shorts pipeline from a highlight extractor into an editorial short-form editor.

The original V2 pipeline is retained as the fallback path. V3 adds structured editorial reasoning, truth-preserving story restructuring, typed edit plans, semantic effects, originality scoring and QA.

## Separation from `ai-shorts`

Development lives on the dedicated Git branch:

```text
ai-shorts-v3
```

Clone that branch into a separate local directory such as `F:\ai-shorts-v3`. Do not merge it into the production `F:\ai-shorts` checkout merely to test V3.

## Architecture discovered and reused

V3 reuses the working V2 components instead of rewriting them:

- Whisper transcription and word timestamps;
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
- word grouping and karaoke;
- FFmpeg/NVENC rendering;
- JSON/debug outputs and logging.

## Pipeline

```text
SOURCE VIDEO
    ↓
Whisper + Word Timestamps
    ↓
Candidate Generator
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
    ├─ Why is this interesting?
    ├─ Editorial Angle
    ├─ Hook V3 candidates
    ├─ Story proposal
    └─ Semantic edit events
    ↓
Truth + schema validation
    ↓
Story Restructurer
    ↓
Typed EditPlan
    ↓
Originality Score + QA
    ├─ pass
    └─ regenerate with weaknesses (bounded retries)
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
FINAL VERTICAL SHORT
```

If any V3 enhancement fails, the system keeps a V2-compatible plan and continues.

## V3 data models

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

Gemini does not generate FFmpeg commands. It produces structured editorial intent; Python validates the intent and the executor translates supported instructions into deterministic editing operations.

## Originality Engine

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

`NO TRANSFORMATION NEEDED` is a valid outcome. A strong standalone source moment is not forced to use extra effects just to inflate an originality number.

## Editorial Angle

The angle is content-derived rather than a fixed category. Examples include close call, unexpected failure, conflict, reveal, reaction, mistake, escalation or payoff.

The structured angle carries:

```text
primary_angle
secondary_angle
viewer_question
stakes
payoff
confidence
```

This information feeds Hook V3, Story Restructurer and Edit Planner.

## Hook V3

Hook Optimizer V2 remains responsible for finding a strong native START. V3 compares additional editorial opening strategies.

### NATIVE

Uses the existing Hook Optimizer start.

### RECONSTRUCTED

Uses a real source quote/reaction from later in the analyzed context as a cold open.

Safety rules:

- the quote must exist in the source transcript;
- `source_start/source_end` must identify the real source audio;
- it must remain inside the context Gemini actually analyzed;
- V3 never synthesizes a fake quote.

### EDITORIAL

Uses concise on-screen framing text. It must be contextually supported and is not allowed to invent high-risk factual anchors such as money amounts, relationships, final/last attempts, wins/losses, bans, deaths or creator intentions when those anchors are absent from the source transcript.

Generic formulas such as `YOU WON'T BELIEVE THIS` and `WAIT UNTIL THE END` are rejected.

A strong native hook is intentionally retained when a transformed alternative is not clearly better.

## Content Truth

V3 treats truth as a hard constraint, not an originality technique.

Deterministic guards reject:

- reconstructed quotes not found in transcript;
- source ranges outside analyzed context;
- invented numeric claims;
- selected high-risk factual terms unsupported by transcript;
- duplicate source ranges unless used explicitly as cold open/replay/callback.

The Gemini prompt also requires neutral wording when uncertain.

## Story Restructurer

Files:

```text
src/story/restructurer.py
```

Target editorial grammar:

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

The system may reorder source segments only when the proposed source ranges pass validation. It does not sort the final plan back into chronological source order.

The existing caption engine already remaps word timestamps in the supplied segment order, so reconstructed timelines remain compatible with captions and karaoke.

### Playback rate

V3 currently forces story segments to `1.0x` even if Gemini suggests another playback rate. This is deliberate: audio, Whisper word timestamps and karaoke are kept synchronized until a complete retiming layer is implemented.

## Edit Planner

File:

```text
src/editing/planner.py
```

Principle:

```text
AI decides WHAT should happen.
Python/FFmpeg decides HOW it happens.
```

The planner combines:

- selected HookPlan;
- editorial angle;
- validated source timeline;
- context overlays;
- semantic visual events;
- audio events;
- caption emphasis metadata;
- content/editorial profile metadata.

## SmartCut 3.0

File:

```text
src/editing/smartcut_v3.py
```

SmartCut 3.0 reuses existing SmartCut boundary logic on each editorial source segment individually, then concatenates those ranges in editorial order.

This is intentionally different from the V2 multi-segment cleanup path, which is optimized for chronological extractive cuts and could otherwise destroy a reconstructed cold open.

SmartCut 3.0 keeps the following V2 strengths:

- word/sentence boundary refinement;
- dead-air cleanup;
- payoff/reaction-aware boundary logic;
- FFmpeg/NVENC export;
- safe fallback on analysis failure.

## Semantic Visual Editing

Files:

```text
src/editing/visual_grammar.py
src/editing/post_processor.py
```

Effects are event-driven, never timer-driven.

Examples:

```text
surprise / payoff  → punch_in
reaction           → face_zoom style punch-in
important detail   → focus_crop style punch-in
```

The deterministic executor currently supports safe center punch-ins for:

- `punch_in`
- `face_zoom`
- `focus_crop`

Runtime crop changes are implemented through FFmpeg `sendcmd` commands on a named crop filter. Unsupported planned effects are ignored safely rather than translated into arbitrary filters.

### Effect budget

`visual_grammar.py` limits effects by semantic event intensity and global plan density. Podcast/dialogue content receives an even smaller visual budget.

The design target is:

```text
intentional > hyperactive
```

## Context Overlays

Context overlays use a separate ASS layer in the V3 post-processor.

They are intended to:

- clarify context;
- establish stakes;
- reduce source dependency;
- make the Short understandable without the long-form source.

Overlay text runs through the same high-risk content-truth guard as editorial hooks.

## Sound Design

Files:

```text
src/editing/sound_design.py
src/editing/post_processor.py
```

Sound design is optional. Audio events have conservative duration/intensity/gain bounds.

Optional local assets live under:

```text
assets/sfx/
```

If an asset is missing, V3 logs a warning and continues without it.

Applied SFX are mixed with the source audio and passed through an output limiter. The current implementation intentionally avoids aggressive automatic loudness processing.

## Existing SmartCrop remains authoritative

V3 does not replace SmartCrop. It preserves:

- face/person tracking;
- gameplay target tracking;
- Reading Focus;
- robust webcam detection;
- `GAMEPLAY_WEBCAM_STACK`;
- forced `--smartcrop-mode gameplay-webcam`;
- blurred panel fill;
- dead zones, smoothing, hysteresis and anti-ping-pong behavior.

## Originality Score

The deterministic score is not intended to estimate platform-detection bypass. It estimates whether the resulting plan adds meaningful editorial value and works independently.

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

Default interpretation:

```text
< 60   → QA requests regeneration
60–75  → acceptable
75–85  → strong
85+    → highly transformed
```

A strong `NO TRANSFORMATION NEEDED` plan can pass below 60 when its native hook and context independence are already high.

## Originality QA and regeneration

File:

```text
src/editing/originality_qa.py
```

QA checks:

- timeline validity;
- hook quality;
- context independence;
- effect budget;
- audio safety;
- payoff identification;
- originality threshold.

If QA fails, the weaknesses are fed back into V3 editorial reasoning. Retries are bounded by `MAX_EDIT_PLAN_RETRIES`. If the target is still not reached, the best valid plan is used with a warning.

## Editorial brand profile

Default profile:

```text
config/editorial_profile.json
```

The supplied `mazclips` profile describes:

- natural/high-information hooks;
- existing karaoke caption style;
- subtle semantic zooms;
- restrained sound design;
- hard cuts;
- short context overlays;
- medium edit density;
- content-specific behavior for gaming, podcast, reaction and entertainment.

## Configuration

Main V3 environment flags:

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

V2 flags continue to work.

## Debug artifacts

For every Short:

```text
output/debug/<video>/clip_<N>/
├── highlight.json
├── hook_analysis.json
├── editorial_angle.json
├── originality_analysis.json
├── edit_plan.json
└── originality_qa.json
```

The authoritative execution plan is also written to:

```text
highlights/v3/<video>/clip_<N>/edit_plan.json
```

## How to create the separate local project

From PowerShell:

```powershell
git clone --branch ai-shorts-v3 --single-branch https://github.com/mazz-dev207/tool-ai-for-yt-shorts.git F:\ai-shorts-v3
cd F:\ai-shorts-v3
```

Create/activate a dedicated virtual environment, or copy only your local `.env` values intentionally. Do not copy caches/output from the V2 project as V3 uses its own project-relative directories.

## How to run

V3 enabled:

```powershell
python main.py video1 --highlight-mode legacy --content-profile gaming --smartcrop-mode gameplay-webcam --v3-mode on
```

Use V3 config/default:

```powershell
python main.py video1 --highlight-mode legacy --content-profile gaming --smartcrop-mode gameplay-webcam
```

Force the V2 fallback pipeline from the V3 checkout:

```powershell
python main.py video1 --highlight-mode legacy --content-profile gaming --smartcrop-mode gameplay-webcam --v3-mode off
```

## Tests

Run V3-focused tests:

```powershell
python -m unittest discover -s tests -p "test_*v3*.py" -v
```

Then the entire inherited + V3 suite:

```powershell
python -m unittest discover -s tests -v
```

Tests do not require real Gemini calls for deterministic components.

## Fallback behavior

V3 enhancements are fail-safe:

```text
V3 Editorial Reasoner fails
→ preserve V2 Hook Optimizer result
→ preserve V2 timeline
→ SmartCut/SmartCrop/Captions/Render continue
```

Missing sound assets or semantic post-processing errors similarly keep the already-rendered V2-compatible vertical clip.

## Known limitations of this V3 implementation

1. Semantic punch-ins and context overlays currently run as a safe post-render pass. The existing caption renderer remains untouched, which minimizes regression risk but means a V3 punch-in can also visually scale already-burned subtitles.
2. `freeze_frame` can be represented/validated in an EditPlan but is not executed automatically by the first deterministic post-processor. Story-level replay is supported through duplicated validated source segments.
3. Playback-rate/speed-ramp execution is intentionally disabled until audio and word timestamps can be retimed together.
4. Context truth validation combines Gemini instructions with conservative deterministic guards; full semantic entailment is not guaranteed. High-risk factual anchors are deliberately rejected when unsupported.
5. Optional SFX require user-supplied local audio assets in `assets/sfx/`.
6. Final visual quality still requires real-video evaluation; unit tests validate deterministic logic, not subjective editorial taste.

## Design principle

For every transformation V3 asks:

```text
Does this improve comprehension?
Does this increase stakes?
Does this strengthen curiosity?
Does this accelerate setup?
Does this preserve payoff?
Does this make the clip work independently?
```

If the answer is no, the edit is not justified.

The goal is not `more effects = more original`.

The goal is:

```text
find the story
→ understand the story
→ identify why viewers should care
→ build the hook
→ restructure truthfully
→ generate a typed EditPlan
→ execute deterministic semantic edits
→ validate originality
→ render a high-quality Short
```

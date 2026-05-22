# Phase 3: AI Improvement - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-22
**Phase:** 3-AI Improvement
**Areas discussed:** Claude integration point, New class strategy, Severity assignment, Autonomous agent loop

---

## Claude Integration Point

| Option | Description | Selected |
|--------|-------------|----------|
| Separate /ai-analysis endpoint | POST /ai-analysis runs YOLO internally then Claude enrichment. Existing /detect unchanged. | ✓ |
| Transparent augmentation on /detect | Calling /detect automatically runs Claude post-processing. Coupled latency. | |

**User's choice:** Separate /ai-analysis endpoint

| Follow-up: AI output schema | | |
|--------|-------------|----------|
| Severity label per detection | Claude assigns minor/moderate/severe | ✓ |
| Class corrections | Claude can re-label YOLO classes | ✓ |
| Natural language damage summary | Short paragraph describing overall damage | |
| Confidence override | Claude adjusts YOLO confidence scores | ✓ |

**User's choice:** Severity label + Class corrections + Confidence override

| Follow-up: API key UX | | |
|--------|-------------|----------|
| In-UI settings panel | Key stored in localStorage, sent per-request as header | ✓ |
| Server .env only | Operator sets key server-side | |
| Per-request header (developer mode) | No UI panel | |

**User's choice:** In-UI settings panel

| Follow-up: Providers | | |
|--------|-------------|----------|
| Claude (Anthropic) | claude-haiku-4-5 recommended | ✓ |
| Gemini (Google) | gemini-1.5-flash recommended | ✓ |
| OpenAI (GPT-4o) | gpt-4o-mini recommended | ✓ |

**Notes:** User asked about "nanobanana" — clarified as Gemini Nano (on-device model, not available via API key; does NOT apply here). User also asked which model is good for vision — recommended haiku/flash/4o-mini tier for cost/speed on damage analysis.

---

## New Class Strategy

| Option | Description | Selected |
|--------|-------------|----------|
| Claude detects new classes from image | Claude identifies scratches/dents/rust/broken glass | |
| Retrain YOLO with new labeled data | Fine-tune best.pt on expanded dataset | |
| Hybrid: YOLO for zones, Claude for damage types | YOLO localizes, Claude classifies within regions | |

**User's initial response (free-text):** "I want this AI to detect any more damages that YOLO have missed or deleting damages that are false positive by YOLO model, same classes only"

**Clarification follow-up:**

| Option | Description | Selected |
|--------|-------------|----------|
| Replace AI-02 with quality improvement | Claude improves quality of existing 11 classes (miss detection + FP removal) | ✓ |
| Both — quality improvement AND new damage types | Quality filter plus new class vocabulary | |

**User's choice:** Replace AI-02 with quality improvement (same 11 classes)

| Follow-up: False positive UX | | |
|--------|-------------|----------|
| Auto-remove silently | Claude's output replaces YOLO annotations | |
| Flag for review | Flagged boxes shown with warning indicator; user approves/dismisses | ✓ |
| Show diff view | Before/after panel of removed/added boxes | |

**User's choice:** Flag for review

**Notes:** This decision significantly redefines AI-02 from "add new class names" to "improve quality of existing classes." The original AI-02 requirement (scratches, dents, rust, broken glass as named classes) is deferred.

---

## Severity Assignment

| Option | Description | Selected |
|--------|-------------|----------|
| Claude assigns severity from visual analysis | Claude analyzes image crop, assigns minor/moderate/severe | |
| Formula: area × confidence → tier | Pure algorithmic, fast, no extra API call | |
| Formula baseline + Claude for borderline cases | Formula first, ±10% boundary cases go to Claude | ✓ |

**User's choice:** Formula baseline + Claude for borderline cases

| Follow-up: Severity display | | |
|--------|-------------|----------|
| Canvas label ("damaged-door [moderate]") | Label on bounding box | ✓ |
| Color-coded box border | minor=yellow, moderate=orange, severe=red | ✓ |
| Session summary panel | Sidebar showing severity breakdown across session | ✓ |
| ZIP export metadata | severity field in annotation JSON | ✓ |

**User's choice:** All four display options

**Notes:** No further questions — user was ready to move on after all four display options selected.

---

## Autonomous Agent Loop

| Third agentic feature options | | |
|--------|-------------|----------|
| Session report agent | Structured damage assessment report across whole session | |
| Batch auto-analyze agent | "Auto-analyze all" button runs pipeline on all images | ✓ |
| Re-annotation suggestion agent | Proposes threshold adjustments for user approval | |

**User's choice:** Batch auto-analyze agent

**Notes:** User asked "from previous two AI features, what should ideally be the 3rd?" — clarified existing two features (Claude orchestrator + provider-agnostic layer) and offered three complementary candidates.

| Follow-up: Batch UX | | |
|--------|-------------|----------|
| Progress bar with image count | "Analyzing 3 of 7 images..." | |
| Per-image status in image list | Spinner/checkmark on each thumbnail | ✓ |
| Silent background + notification | Nothing shown during processing | |

**User's choice:** Per-image status in image list

| Follow-up: Concurrency | | |
|--------|-------------|----------|
| Non-blocking (Recommended) | Processed images available immediately | ✓ |
| Lock UI until all done | UI locked until full batch completes | |

**User's choice:** Non-blocking

---

## Claude's Discretion

- Exact prompt templates for AI analysis (vision prompt content)
- Provider interface abstraction layer design (unified Claude/Gemini/OpenAI SDK wrapper)
- Formula threshold values for severity tier boundaries (pixel area cutoffs)
- Error handling and fallback when AI provider call fails (bad key, rate limit)
- Fine-tuning script dependencies and dataset format (AI-01 implementation details)

## Deferred Ideas

- **Session report agent** — generate structured damage report across full session (natural v2 after batch baseline works)
- **Iterative re-annotation loop** — YOLO + AI refine until result stabilizes (latency tradeoff; defer)
- **Gemini Nano** — on-device, not API-accessible; revisit if Google opens the API
- **AI-02 expanded class vocabulary** — scratches/dents/rust/broken glass as named YOLO classes; requires labeled dataset; deferred to v2

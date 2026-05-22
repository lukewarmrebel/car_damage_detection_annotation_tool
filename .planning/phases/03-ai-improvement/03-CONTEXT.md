# Phase 3: AI Improvement - Context

**Gathered:** 2026-05-22
**Status:** Ready for planning

<domain>
## Phase Boundary

Upgrade the AI detection layer: add a provider-agnostic AI orchestration endpoint (Claude/Gemini/OpenAI) that post-processes YOLO results to improve quality (catch misses, flag false positives) and assign severity labels. Add a batch auto-analyze feature that runs the full YOLO + AI pipeline across all uploaded images in one user-triggered action. Implement a reproducible fine-tuning pipeline script. The existing `/detect` endpoint and annotation canvas remain unchanged.

</domain>

<decisions>
## Implementation Decisions

### AI Integration Endpoint
- **D-01:** Add a separate `/ai-analysis` endpoint — does NOT modify the existing `/detect` endpoint. Existing detection flow is backwards-compatible.
- **D-02:** The `/ai-analysis` endpoint runs YOLO internally on the session image, then passes YOLO results + image to the user-selected AI provider for enrichment.
- **D-03:** AI provider is user-selectable: Claude (Anthropic), Gemini (Google), OpenAI (GPT-4o). Support all three via a unified provider interface.
- **D-04:** Recommended models per provider: `claude-haiku-4-5` (Claude), `gemini-1.5-flash` (Gemini), `gpt-4o-mini` (OpenAI). Let users pick model within their chosen provider.
- **D-05:** API key is supplied by the user via an in-UI settings panel (not hardcoded server-side). Key is stored in browser `localStorage` and sent per-request as a request header. No server-side key storage.

### AI Output Schema
- **D-06:** AI enrichment returns three enrichments per detection: `severity` (new field — minor/moderate/severe), `class_name` override (AI can correct YOLO's class label), `confidence` override (AI can adjust YOLO's confidence score).
- **D-07:** Enriched annotations add a `severity` field and an `ai_source` field (provider name + model) to the existing `Annotation` model.

### Quality Improvement (replaces AI-02 "expanded classes")
- **D-08:** AI improves quality of the existing 11 YOLO classes — does NOT add new class names. The class vocabulary stays at 11 classes.
- **D-09:** AI performs two quality tasks: (a) identifies likely missed detections (false negatives) and adds them as new annotation candidates; (b) flags likely false positives from YOLO with a warning indicator.
- **D-10:** Flagged false positives are NOT auto-removed. They appear with a distinct warning color/badge in the UI. User approves (keep) or dismisses (remove) each flagged detection. User stays in control.

### Severity Assignment (AI-03)
- **D-11:** Hybrid severity: formula runs first — `box_pixel_area × YOLO_confidence → tier threshold`. Cases within ±10% of a tier boundary are escalated to the AI provider for visual reasoning.
- **D-12:** Severity tiers: minor / moderate / severe.
- **D-13:** Severity display — ALL of the following:
  - Canvas label format: `"damaged-door [moderate]"` (class + severity on bounding box)
  - Color-coded box border: minor=yellow, moderate=orange, severe=red
  - Session summary sidebar panel: total detections grouped by severity tier across all images
  - ZIP export metadata: `severity` field included in annotation JSON

### Batch Auto-Analyze Agent (3rd agentic feature)
- **D-14:** "Auto-analyze all" button in the UI runs the full YOLO + AI pipeline on every image in the current session sequentially — user does not need to click "Analyze" per image.
- **D-15:** Progress display: per-image spinner/checkmark shown on each image thumbnail in the sidebar while batch is running.
- **D-16:** Non-blocking: images that have completed analysis are immediately available for user review and editing while remaining images are still being processed.

### Fine-Tuning Pipeline (AI-01)
- **D-17:** Fine-tuning pipeline is a standalone CLI Python script (`finetune.py`) that accepts a labeled dataset path and produces a new `best.pt`. Accompanied by a `FINE_TUNING.md` doc explaining the workflow.
- **D-18:** Script is NOT integrated into the web UI — it is a developer/operator tool run from the terminal.

### Claude's Discretion
- Exact prompt templates for AI analysis (what the vision prompt says to Claude/Gemini/OpenAI)
- Provider interface abstraction layer design (how to unify Claude SDK / Gemini SDK / OpenAI SDK)
- Formula threshold values for severity tier boundaries (exact pixel area cutoffs)
- Error handling and fallback when AI provider call fails (e.g., bad key, rate limit)
- Fine-tuning script dependencies and dataset format

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Existing Backend
- `backend/app.py` — FastAPI app; `detect_session()` function, `Annotation` model, `ImageSession` model, `sessions` dict. New `/ai-analysis` endpoint goes here.
- `backend/__init__.py` — Package init (currently empty).

### Existing Frontend
- `static/app.js` — Plain JS frontend; `state` object, `setStatus()`, `activeImage()`. Batch agent, settings panel, and severity display integrate here.
- `static/index.html` — HTML structure; where settings panel and batch button UI markup goes.
- `static/styles.css` — CSS; severity color coding and flagged detection styles go here.

### Requirements
- `.planning/REQUIREMENTS.md` — AI-01, AI-02, AI-03 (note: AI-02 redefined during discuss-phase — see D-08)
- `.planning/ROADMAP.md` — Phase 3 success criteria

### Model
- `best.pt` — Current YOLOv8 model (11 classes) in repo root; fine-tuning starts from this
- `class_dict.txt` — 11 YOLO class names; scope of quality improvement is these classes only

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `Annotation` model (`backend/app.py:37`) — add `severity: str | None = None` and `ai_source: str | None = None` fields; existing fields (`class_name`, `confidence`) are override targets for AI enrichment
- `detect_session()` (`backend/app.py:247`) — AI analysis endpoint reuses this internally then layers enrichment on top
- `sessions` dict (`backend/app.py:105`) — AI analysis stores enriched results back here, same session lifecycle
- `state.yoloProcessed` Set in `static/app.js:44` — extend pattern for tracking `aiProcessed` per image

### Established Patterns
- Provider model follows existing YOLO path: lazy-load client on first use, raise `HTTPException(500)` if not configured
- Color convention: backend works in BGR; frontend receives JPEG bytes — severity color coding lives in CSS/JS, not OpenCV
- Session storage is in-memory; no persistence across server restarts — AI enrichment inherits this constraint

### Integration Points
- New `/ai-analysis` endpoint returns same `DetectResponse` shape (or extended version) so frontend can reuse existing annotation rendering
- Settings panel (new) stores API key in `localStorage`; sends key as `X-AI-Api-Key` header with each `/ai-analysis` call
- Batch loop in frontend calls `/ai-analysis` for each `image_id` in `state.images`, sequentially, updating per-image status

</code_context>

<specifics>
## Specific Ideas

- "Auto-analyze all" button should be in the sidebar tools section alongside the existing "Detect" button
- Severity color scheme: yellow / orange / red (minor / moderate / severe) — intuitive traffic-light style
- Settings panel UI: provider dropdown (Claude / Gemini / OpenAI), model selector, API key text input (password type), save button
- The session summary panel showing severity breakdown is a new sidebar section below the image list

</specifics>

<deferred>
## Deferred Ideas

- **Session report agent** — After all images are analyzed, generate a structured damage assessment report (damage count by severity, most affected zones, export with ZIP). Natural v2 feature once batch auto-analyze works.
- **Iterative re-annotation loop** — YOLO + AI refine cycle until result count stabilizes. More thorough but adds latency; defer until batch baseline is validated.
- **Gemini Nano / on-device inference** — Not available via standard Gemini API; revisit if Google opens on-device API in future.
- **AI-02 expanded class vocabulary** (scratches, dents, rust, broken glass as named classes) — replaced by quality improvement approach (D-08). Could be revisited if a labeled dataset is assembled for a v2 YOLO fine-tune.

</deferred>

---

*Phase: 3-AI Improvement*
*Context gathered: 2026-05-22*

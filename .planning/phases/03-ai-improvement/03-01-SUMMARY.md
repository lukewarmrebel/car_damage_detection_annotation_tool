---
plan: 03-01
phase: 03-ai-improvement
status: complete
completed: 2026-05-22
subsystem: ai-analysis
tags: [ai, severity, backend, frontend, yolo, batch]
dependencies:
  requires: []
  provides: [POST /ai-analysis, severity-rendering, settings-panel, batch-analyze, zip-metadata-sidecar]
  affects: [backend/app.py, static/app.js, static/index.html, static/styles.css, requirements.txt]
tech_stack:
  added: [anthropic, google-generativeai, openai]
  patterns: [provider-dispatch, severity-formula, graceful-degradation, localStorage-key-management]
key_files:
  created: []
  modified:
    - backend/app.py
    - static/app.js
    - static/index.html
    - static/styles.css
    - requirements.txt
decisions:
  - "AI provider dispatch is stateless: API key passed per-request via X-AI-Api-Key header, never stored server-side (D-05)"
  - "severity_from_box() formula-first: pixel_area * confidence tiered at 15000/60000; boundary cases within 10% escalated to AI (D-11)"
  - "Gemini SDK requires inline_data wrapper with base64-encoded image bytes, not raw bytes dict"
  - "Settings localStorage keys are provider-scoped: ai_api_key_{provider} so switching providers does not clobber other keys"
  - "Gemini model list updated to 2.5-generation: gemini-2.5-flash, gemini-2.5-flash-lite, gemini-1.5-flash"
metrics:
  duration_minutes: ~180
  tasks_completed: 2
  files_modified: 5
---

# Phase 3 Plan 01: AI Analysis Endpoint + Severity + Batch UI Summary

## One-liner

Multi-provider AI enrichment layer (Claude/Gemini/OpenAI) over YOLO detections with formula-first severity tiers, severity-colored canvas rendering, settings panel, batch auto-analyze, and ZIP metadata sidecars.

## What Was Built

**Backend (`backend/app.py`):**
- Extended the `Annotation` Pydantic model with two optional fields: `severity: str | None` (minor/moderate/severe) and `ai_source: str | None` (claude/gemini/openai).
- `severity_from_box()` helper: computes `pixel_area * confidence` score and maps to minor/moderate/severe tiers at thresholds 15000 and 60000; marks boundary cases (within 10% of a threshold) as `is_boundary=True` for AI escalation.
- `build_ai_prompt()`: constructs a structured prompt instructing the AI provider to review YOLO detections, confirm/override severity for boundary cases, identify missed damage regions, and flag false positives — all constrained to the 11 valid class names.
- Three provider dispatch functions: `call_claude()`, `call_gemini()`, `call_openai()` — each accepts `(prompt, image_bytes, model, api_key)`, encodes the image as base64, and calls the respective SDK. All raise `HTTPException(502)` on SDK errors.
- `POST /ai-analysis` endpoint: validates `image_id`, reads `X-AI-Api-Key` header (401 if missing), calls `detect_session()` internally for fresh YOLO detections, applies formula severity, dispatches to AI provider, merges enriched results (`enriched` overrides, `missed` added as new annotations with `source="ai-missed"`, `false_positive_indices` sets `source="ai-fp"`), updates `session.annotations`, returns `DetectResponse`. Graceful degradation: if AI JSON parse fails, returns formula-only severity without crashing.
- Updated `/export` to write a per-image `_metadata.json` sidecar in the ZIP with: `image_id`, `filename`, and per-annotation `id/class_name/confidence/severity/source/ai_source`.

**Frontend (`static/app.js`):**
- `state.aiProcessed`: new `Set<string>` tracking which image_ids have been AI-analyzed.
- `severityColor()`: maps severity string to RGB triple for canvas rendering.
- `drawAnnotation()` updated: severity-colored border re-strokes rectangle boxes; label text extended with `[severity]` tag and `[?FP]` tag for false-positive-flagged annotations.
- `loadSettings()` / `saveSettings()` / `updateModelOptions()`: read/write `ai_provider`, `ai_api_key_{provider}`, `ai_model_{provider}` keys in localStorage; `updateModelOptions()` populates model dropdown per provider.
- `runAIAnalysis()`: single-image AI analysis — reads provider/key from localStorage, sends `POST /ai-analysis` with proper headers, merges result into `state.images`, calls `draw()` and `refreshSessionSummary()`.
- `runBatchAnalysis()` / `setBatchStatus()`: sequential batch over all images; thumbnail badge shows spinner during processing, checkmark on completion, `!` on error.
- `refreshSessionSummary()`: aggregates annotation severity counts across all images and renders them in the `#sessionSummary` div.
- `loadSettings()` called on page load so settings persist across refreshes.
- Event listeners wired for all new buttons and the provider change event.

**Frontend (`static/index.html`):**
- Settings button added to top-actions bar.
- "AI Analyze" and "Auto-analyze all" buttons added to Tools section.
- Session Summary sidebar section with `#sessionSummary` div.
- `<dialog id="settingsDialog">` with provider dropdown, model select, password input for API key, Save and Close buttons.

**Frontend (`static/styles.css`):**
- CSS variables `--severity-minor`, `--severity-moderate`, `--severity-severe`, `--severity-fp` added to `:root`.
- Classes: `.summary-panel`, `.summary-row`, `.severity-minor/moderate/severe`, `.muted-text`, `.dialog-actions`, `.batch-spinner`, `.batch-done`, `@keyframes spin`.

**Dependencies (`requirements.txt`):**
- Added `anthropic`, `google-generativeai`, `openai`.

## Files Modified

- `backend/app.py` — Annotation model extended; severity_from_box, build_ai_prompt, call_claude, call_gemini, call_openai, POST /ai-analysis added; /export updated with metadata sidecar
- `requirements.txt` — anthropic, google-generativeai, openai added
- `static/app.js` — state.aiProcessed, severityColor, updated drawAnnotation, loadSettings, saveSettings, updateModelOptions, runAIAnalysis, runBatchAnalysis, setBatchStatus, refreshSessionSummary, event listeners, loadSettings on page load
- `static/index.html` — Settings button, AI Analyze button, Auto-analyze all button, Session Summary section, settings dialog
- `static/styles.css` — severity color variables, summary panel and badge classes, batch spinner animation

## Key Decisions Made During Execution

1. **Gemini inline_data format:** Initial implementation sent raw bytes in the wrong dict structure. Fixed to use `{"inline_data": {"mime_type": "image/jpeg", "data": base64_encoded_string}}` as required by the google-generativeai SDK.
2. **HTTPException re-raise in provider dispatch:** The original implementation silently swallowed provider errors by catching them before the HTTPException could propagate. Fixed to re-raise HTTPExceptions so 502 errors surface correctly to the caller.
3. **localStorage key read in runAIAnalysis / runBatchAnalysis:** Both functions were reading the model from the DOM element value (stale if settings changed without re-opening the dialog). Fixed to read `localStorage.getItem("ai_model_" + provider)` so the persisted value is used.
4. **loadSettings on page load:** Without this, the provider dropdown showed the default value on first load even if the user had previously saved a different provider. Added `loadSettings()` call at script initialization.
5. **Gemini model list:** Updated from the plan's `gemini-1.5-flash` single entry to include `gemini-2.5-flash`, `gemini-2.5-flash-lite`, and `gemini-1.5-flash` to align with currently available Gemini models.

## Verification Results

Human-verified by user: AI analysis endpoint works end-to-end with the Gemini provider. Severity labels render on bounding boxes with color-coded borders. Settings panel saves and loads provider and API key correctly. Checkpoint approved after post-checkpoint bug fixes were applied.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed Gemini inline_data image encoding**
- **Found during:** Post-checkpoint verification
- **Issue:** `call_gemini()` was constructing the image part as `{"mime_type": ..., "data": raw_bytes}` at the top level, causing the Gemini SDK to reject it with an invalid argument error.
- **Fix:** Wrapped in `{"inline_data": {"mime_type": "image/jpeg", "data": base64_string}}` per SDK requirements.
- **Files modified:** `backend/app.py`

**2. [Rule 1 - Bug] Fixed silent swallowing of provider HTTPExceptions**
- **Found during:** Post-checkpoint verification
- **Issue:** The `except Exception` block in provider dispatch was catching `HTTPException` before it could propagate, converting 502 errors into silent no-ops.
- **Fix:** Added `except HTTPException: raise` before the generic `except Exception` block in all three provider functions.
- **Files modified:** `backend/app.py`

**3. [Rule 1 - Bug] Fixed model read using stale DOM value in runAIAnalysis/runBatchAnalysis**
- **Found during:** Post-checkpoint verification
- **Issue:** Both functions read `aiModelSelect.value` (DOM) rather than localStorage, so the model selection reverted to the DOM default on page reload.
- **Fix:** Changed both to read `localStorage.getItem("ai_model_" + provider)` with a fallback to provider default.
- **Files modified:** `static/app.js`

**4. [Rule 2 - Missing critical functionality] Added loadSettings() on page load**
- **Found during:** Post-checkpoint verification
- **Issue:** Settings dialog pre-populated correctly on re-open, but the provider and model state was not initialized at page load, so `runAIAnalysis` would use stale defaults until the settings dialog was opened.
- **Fix:** Added `loadSettings()` call at end of script initialization.
- **Files modified:** `static/app.js`

**5. [Rule 1 - Bug] Updated Gemini model list to current 2.5-generation models**
- **Found during:** Post-checkpoint testing
- **Issue:** Plan specified `gemini-1.5-flash` only; attempting to use `gemini-2.5-flash` (a current model) would fail silently as it was not in the dropdown.
- **Fix:** Updated `updateModelOptions()` to list `gemini-2.5-flash`, `gemini-2.5-flash-lite`, and `gemini-1.5-flash`.
- **Files modified:** `static/app.js`

## Known Stubs

None — all severity data is live from the AI provider response; no hardcoded placeholder values flow to the UI.

## Threat Flags

No new security-relevant surface introduced beyond what was modeled in the plan's threat register. The `/ai-analysis` endpoint matches the documented trust boundaries (T-03-01 through T-03-SC). API key header validation (401 on missing key) and class_name validation against VALID_CLASSES are implemented per T-03-02 and T-03-06.

## Self-Check: PASSED

- `backend/app.py` — exists and contains `/ai-analysis` endpoint, `severity_from_box`, `Annotation.severity`, `Annotation.ai_source`
- `static/app.js` — contains `runAIAnalysis`, `runBatchAnalysis`, `refreshSessionSummary`, `loadSettings`, `severityColor`, `state.aiProcessed`
- `static/index.html` — contains `aiAnalyzeBtn`, `autoAnalyzeBtn`, `settingsBtn`, `settingsDialog`, `sessionSummary`
- `static/styles.css` — contains `--severity-minor`, `.summary-panel`, `.batch-spinner`
- `requirements.txt` — contains `anthropic`, `google-generativeai`, `openai`

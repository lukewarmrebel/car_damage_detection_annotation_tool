# Roadmap: Car Damage Detection & Annotation Tool (Web)

## Overview

A brownfield web rewrite of an existing Python/Tkinter desktop app. Phase 1 stands up the FastAPI backend with all API endpoints and a served frontend shell — the backend is the foundation everything else depends on. Phase 2 delivers the full browser annotation experience wired to Phase 1's APIs, completing the core user workflow. Phase 3 upgrades the AI layer with a provider-agnostic enrichment endpoint, severity labels, batch auto-analyze, and a reproducible fine-tuning pipeline. Phases 2 and 3 can proceed in parallel once Phase 1 ships.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 1: Backend API** - FastAPI backend serves detection, annotation, ZIP export, and the frontend shell
- [x] **Phase 2: Browser UI** - Full annotation canvas wired to backend APIs — upload, detect, draw, export
- [x] **Phase 3: AI Improvement** - Provider-agnostic AI enrichment endpoint, severity labels, batch auto-analyze, and fine-tuning pipeline

## Phase Details

### Phase 1: Backend API
**Goal**: A running FastAPI server that accepts image uploads, returns YOLO detections, applies annotation edits, and exports ZIPs — with the frontend shell served at the root route
**Mode:** mvp
**Depends on**: Nothing (first phase)
**Requirements**: BACK-01, BACK-02, BACK-03, BACK-04
**Success Criteria** (what must be TRUE):
  1. User can POST an image to `/detect` and receive JSON with bounding boxes, class names, and confidence scores from the existing YOLOv8 model
  2. User can POST annotation edits (shapes to add or remove) and receive the annotated image as a response
  3. User can GET `/export` and download a ZIP containing all annotated images accumulated in the session
  4. Navigating to the server root in a browser returns a valid HTML page (frontend shell placeholder)
**Plans**: 2 plans

Plans:
- [x] 01-01-PLAN.md — FastAPI project setup, YOLO detection endpoint, session management
- [x] 01-02-PLAN.md — Annotation edit endpoint, ZIP export endpoint, static file serving

### Phase 2: Browser UI
**Goal**: A fully functional browser annotation interface where users can upload images, trigger AI detection, draw and erase annotations, navigate a multi-image session, and download the ZIP export
**Mode:** mvp
**Depends on**: Phase 1
**Requirements**: FRONT-01, FRONT-02, FRONT-03, FRONT-04, FRONT-05, FRONT-06, FRONT-07
**Success Criteria** (what must be TRUE):
  1. User can select one or more image files from disk and see them queued in the browser
  2. User can click "Detect" on any loaded image and see YOLO bounding boxes drawn over the image in the browser
  3. User can draw circle, rectangle, and text annotations directly on the image canvas
  4. User can undo the last annotation, erase a selected annotation, and clear all annotations from an image
  5. User can zoom into and pan around the canvas, and navigate forward/backward through all uploaded images
  6. User can click "Export ZIP" and receive a download of all annotated images
**Plans**: 2 plans

Plans:
- [x] 02-01-PLAN.md — Canvas component with zoom/pan, image navigation, and upload flow
- [x] 02-02-PLAN.md — AI detection overlay, draw tools (circle, rect, text), undo/erase/clear, ZIP download

**UI hint**: yes

### Phase 3: AI Improvement
**Goal**: The detection model surfaces richer damage assessments (severity labels, AI quality enrichment) via a provider-agnostic AI endpoint, and a documented fine-tuning pipeline exists so model quality can be improved reproducibly
**Mode:** mvp
**Depends on**: Phase 1
**Requirements**: AI-01, AI-02, AI-03
**Success Criteria** (what must be TRUE):
  1. User can select an AI provider (Claude/Gemini/OpenAI), configure their API key in the Settings panel, and run AI analysis that enriches existing YOLO detections with severity labels, missed-detection candidates, and false-positive flags
  2. Every AI-enriched bounding box carries a severity label (minor / moderate / severe) shown as a color-coded border and label text; a session summary panel shows aggregate severity counts
  3. A fine-tuning script and accompanying documentation exist; running the script with a labelled dataset produces a new `best.pt` model file
**Plans**: 2 plans

Plans:
- [x] 03-01-PLAN.md — AI analysis endpoint (Claude/Gemini/OpenAI), severity derivation, FP flagging, batch auto-analyze, frontend settings panel and severity display
- [x] 03-02-PLAN.md — Fine-tuning pipeline script (finetune.py) and documentation (FINE_TUNING.md)

## Progress

**Execution Order:**
Phases 2 and 3 can run in parallel after Phase 1 completes (parallelization: true).
Order: 1 → (2 ∥ 3)

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Backend API | 2/2 | Complete | 2026-05-22 |
| 2. Browser UI | 2/2 | Complete | 2026-05-22 |
| 3. AI Improvement | 2/2 | Complete | 2026-05-22 |

# Car Damage Detection & Annotation Tool

## What This Is

A web-based car damage detection and annotation platform. Users upload vehicle images, an AI model (YOLOv8) automatically detects and labels damage regions, and users can review, adjust annotations manually, then export results. Built as a portfolio project showcasing computer vision + web engineering.

## Core Value

Instant AI-powered damage detection with human-in-the-loop review — faster and more accurate than purely manual annotation.

## Requirements

### Validated

- ✓ YOLOv8 damage detection (11 classes) — existing desktop app
- ✓ Manual annotation tools (circle, rectangle, text) — existing desktop app
- ✓ ZIP export of annotated images — existing desktop app

### Active

- [ ] Web frontend: upload → AI detect → annotate → export (no login required)
- [ ] FastAPI backend with detection and annotation endpoints
- [ ] Expanded AI model: more damage classes + severity estimation (minor/moderate/severe)
- [ ] Fine-tuning pipeline for reproducible model improvement
- [ ] Annotation canvas in browser (draw, erase, undo, zoom/pan)
- [ ] Export annotated images as ZIP from web UI

### Out of Scope

- User accounts / authentication — portfolio demo, no multi-user persistence needed
- Case management / claim tracking — out of scope for v1
- Native desktop .exe distribution — switching to web
- PDF report generation — deferred post-v1

## Context

- Existing codebase: Python/Tkinter desktop app (862 lines, single file) with OpenCV, PIL, YOLOv8 ultralytics
- Model: `best.pt` (YOLOv8, 11 damage classes defined in `class_dict.txt`)
- Current merge conflicts resolved; YOLO path fixed; modern dark UI shipped; live preview + keyboard shortcuts added
- PyInstaller build script exists (`build.bat`) but web app is the target
- Stack decision: FastAPI backend (Python, keeps existing ML code), React or plain JS frontend

## Constraints

- **Stack**: Python backend — keeps all existing OpenCV/ultralytics code
- **Model**: Must build on existing `best.pt` as fine-tuning base
- **Deployment**: Single-server web app (not microservices) for portfolio simplicity

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Web app over desktop | Better portfolio showcase, broader accessibility | — Pending |
| FastAPI backend | Stays in Python, async, auto-docs, fast to build | — Pending |
| Fine-tuning pipeline over one-off retrain | Reproducible improvement over time | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd:complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-05-22 after initialization*

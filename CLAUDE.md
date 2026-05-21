# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Application

```bash
pip install -r requirements.txt
python Car_damage_annotation_compression_tool_with_AI_damage_detection.py
```

There are no tests or linters configured for this project.

## Critical: YOLO Model Path

The main script hardcodes the model path at line 19:
```python
model = YOLO(r'C:\Users\ADMIN\Desktop\Office stuff\FT BI tool\Car_damage_annotation_compression_tool\best.pt')
```
This must point to a valid `best.pt` file. The repo includes `best.pt` in the root directory — update the path to a relative or absolute path matching the local machine before running.

## Critical: Unresolved Merge Conflicts

Both `Car_damage_annotation_compression_tool_with_AI_damage_detection.py` and `class_dict.txt` contain unresolved Git merge conflict markers (`<<<<<<< HEAD`, `=======`, `>>>>>>>`). The two conflicting versions are identical in content, so either conflict block can be kept — the markers just need to be removed before Python can parse the file.

## Architecture

The entire application is a single Python script (`Car_damage_annotation_compression_tool_with_AI_damage_detection.py`) with no classes — all state is held in module-level globals and all logic is in top-level functions. The GUI is constructed at the bottom of the file and `root.mainloop()` is the last line.

### Image State Model

Two globals track image state per loaded image:
- `original_img` — the working base image. Filters, rotations, flips, and crops mutate this directly and are not individually undoable.
- `original_img_state` — a snapshot taken when an image is first loaded; used by Reset View to restore the pre-edit state.

Annotations are stored separately in `annotations_dict` (keyed by file path, value is a list of shape dicts) and re-drawn onto a copy of `original_img` on every `update_canvas()` call. This means annotations float on top and are always drawn last.

### Color Convention

OpenCV reads images as BGR; images are immediately converted to RGB on load (`cv2.cvtColor(..., cv2.COLOR_BGR2RGB)`) and remain RGB throughout the app. They are converted back to BGR only when written to disk.

### YOLO Integration

`run_yolo_detection()` reads the image fresh from disk (bypassing any applied filters) and appends bounding box annotations with `"source": "yolo"` and `"class_name"` fields. The `yolo_processed` set prevents re-running YOLO on the same image within a session; clearing markings removes the image from this set. Class names come from `model.names` (11 damage categories defined in `class_dict.txt`).

### Output

Annotated images are written to `processed_images/` (created on startup). ZIP export re-reads each original file, redraws annotations, resizes to max 1024px on the longest side, and compresses at JPEG quality 85.

### `Model versions/` Directory

Contains earlier iterations of the script (without YOLO, with Gemini API). These are historical reference only and not part of the active application.

## GSD Workflow

This project uses the Get Shit Done (GSD) planning framework. Planning artifacts live in `.planning/`.

**Current milestone:** Web rewrite (3 phases)
**Phase 1:** Backend API (FastAPI + YOLO detection + annotation endpoints + ZIP export)
**Phase 2:** Browser UI (canvas annotation, upload, export)
**Phase 3:** AI Improvement (expanded classes, severity labels, fine-tuning pipeline)

**Key planning files:**
- `.planning/PROJECT.md` — project context and decisions
- `.planning/REQUIREMENTS.md` — v1 requirements with REQ-IDs
- `.planning/ROADMAP.md` — phase structure and success criteria
- `.planning/STATE.md` — current position and blockers

**Next step:** `/gsd:plan-phase 1` to generate the Phase 1 execution plan.

**Workflow commands:**
- `/gsd:plan-phase N` — plan a phase
- `/gsd:execute-phase N` — execute a phase plan
- `/gsd:progress` — show current project state

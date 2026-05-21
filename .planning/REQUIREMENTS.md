# Requirements

## v1 Requirements

### Web Backend
- [ ] **BACK-01**: User can POST an image and receive bounding-box detections with class names and confidence scores
- [ ] **BACK-02**: User can POST annotation edits (add/remove shapes) and receive the annotated image
- [ ] **BACK-03**: User can GET a ZIP of all annotated images in a session
- [ ] **BACK-04**: Backend serves the frontend and static assets

### Web Frontend
- [ ] **FRONT-01**: User can upload one or more images from the browser
- [ ] **FRONT-02**: User can trigger AI detection and see bounding boxes overlaid on the image
- [ ] **FRONT-03**: User can draw circle, rectangle, and text annotations on the canvas
- [ ] **FRONT-04**: User can erase, undo, and clear annotations
- [ ] **FRONT-05**: User can zoom and pan the image canvas
- [ ] **FRONT-06**: User can navigate between uploaded images
- [ ] **FRONT-07**: User can download a ZIP of all annotated images

### AI Improvement
- [ ] **AI-01**: Fine-tuning pipeline (script + docs) that takes a labelled dataset and produces a new best.pt
- [ ] **AI-02**: Expanded class set beyond current 11 — includes scratches, dents, rust, broken glass
- [ ] **AI-03**: Each detection includes a severity label (minor / moderate / severe) derived from box area and confidence

## v2 Requirements (Deferred)

- User accounts and persistent case storage
- PDF damage report export
- Batch processing of many images server-side
- Mobile-responsive UI

## Out of Scope

- Authentication / multi-user — portfolio demo, single-session
- Desktop .exe distribution — replaced by web app
- Case / claim management system — beyond v1 scope

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| BACK-01 | Phase 1 | Pending |
| BACK-02 | Phase 1 | Pending |
| BACK-03 | Phase 1 | Pending |
| BACK-04 | Phase 1 | Pending |
| FRONT-01 | Phase 2 | Pending |
| FRONT-02 | Phase 2 | Pending |
| FRONT-03 | Phase 2 | Pending |
| FRONT-04 | Phase 2 | Pending |
| FRONT-05 | Phase 2 | Pending |
| FRONT-06 | Phase 2 | Pending |
| FRONT-07 | Phase 2 | Pending |
| AI-01 | Phase 3 | Pending |
| AI-02 | Phase 3 | Pending |
| AI-03 | Phase 3 | Pending |

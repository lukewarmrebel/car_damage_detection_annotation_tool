from __future__ import annotations

import base64
import io
import json
import os
import tempfile
import uuid
import zipfile
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"
MODEL_PATH = Path(os.getenv("YOLO_MODEL_PATH", BASE_DIR / "best.pt"))
MAX_EXPORT_SIDE = 1024
JPEG_QUALITY = 85

VALID_CLASSES = {
    "damage-front-windscreen",
    "damaged-door",
    "damaged-fender",
    "damaged-front-bumper",
    "damaged-head-light",
    "damaged-hood",
    "damaged-rear-bumper",
    "damaged-rear-window",
    "damaged-side-window",
    "damaged-tail-light",
    "quaterpanel-dent",
}

MINOR_MAX = 15000
MODERATE_MAX = 60000
BOUNDARY_MARGIN = 0.10

app = FastAPI(title="Car Damage Detection Annotation API")

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class Point(BaseModel):
    x: int
    y: int


class Annotation(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: Literal["rectangle", "circle", "text"]
    color: tuple[int, int, int] = (255, 0, 0)
    thickness: int = 2
    start: Point | None = None
    end: Point | None = None
    center: Point | None = None
    radius: int | None = None
    position: Point | None = None
    text: str | None = None
    font_scale: float = 0.7
    font_size: int | None = None
    source: str | None = None
    class_name: str | None = None
    confidence: float | None = None
    severity: str | None = None
    ai_source: str | None = None


class DetectResponse(BaseModel):
    image_id: str
    filename: str
    width: int
    height: int
    detections: list[Annotation]


class ImageUploadResponse(BaseModel):
    image_id: str
    filename: str
    width: int
    height: int


class AnnotationUpdate(BaseModel):
    image_id: str
    annotations: list[Annotation] = []
    remove_ids: list[str] = []
    replace: bool = False


class ImageEditRequest(BaseModel):
    image_id: str
    action: Literal["rotate", "flip", "crop", "filter", "reset"]
    degrees: int | None = None
    axis: Literal["horizontal", "vertical"] | None = None
    start: Point | None = None
    end: Point | None = None
    filter_type: Literal[
        "grayscale",
        "blur",
        "sharpen",
        "edge_detection",
        "contrast",
        "color_thresholding",
        "laplacian",
        "thermal",
        "high_pass",
    ] | None = None


class ImageSession(BaseModel):
    filename: str
    image: bytes
    original_image: bytes
    annotations: list[Annotation] = []


class AIAnalysisRequest(BaseModel):
    image_id: str
    provider: Literal["claude", "gemini", "openai"]
    model: str


model: YOLO | None = None
sessions: dict[str, ImageSession] = {}


def get_model() -> YOLO:
    global model
    if model is None:
        if not MODEL_PATH.exists():
            raise HTTPException(status_code=500, detail=f"YOLO model not found: {MODEL_PATH}")
        model = YOLO(str(MODEL_PATH))
    return model


def decode_image(image_bytes: bytes) -> np.ndarray:
    data = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image")
    return image


def encode_jpeg(image: np.ndarray) -> bytes:
    ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    if not ok:
        raise HTTPException(status_code=500, detail="Could not encode annotated image")
    return buffer.tobytes()


def encode_like_upload(image: np.ndarray, filename: str) -> bytes:
    suffix = Path(filename).suffix.lower()
    ext = ".png" if suffix == ".png" else ".jpg"
    params = [] if ext == ".png" else [int(cv2.IMWRITE_JPEG_QUALITY), 95]
    ok, buffer = cv2.imencode(ext, image, params)
    if not ok:
        raise HTTPException(status_code=500, detail="Could not encode edited image")
    return buffer.tobytes()


def render_annotation(image: np.ndarray, annotation: Annotation) -> None:
    color = annotation.color
    thickness = annotation.thickness

    if annotation.type == "rectangle":
        if not annotation.start or not annotation.end:
            return
        start = (annotation.start.x, annotation.start.y)
        end = (annotation.end.x, annotation.end.y)
        cv2.rectangle(image, start, end, color, thickness)
        if annotation.class_name:
            label_y = max(start[1] - 10, 15)
            cv2.putText(image, annotation.class_name, (max(start[0], 0), label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    elif annotation.type == "circle":
        if not annotation.center or annotation.radius is None:
            return
        cv2.circle(image, (annotation.center.x, annotation.center.y), annotation.radius, color, thickness)
    elif annotation.type == "text":
        if not annotation.position or not annotation.text:
            return
        cv2.putText(
            image,
            annotation.text,
            (annotation.position.x, annotation.position.y),
            cv2.FONT_HERSHEY_SIMPLEX,
            (annotation.font_size / 24) if annotation.font_size else annotation.font_scale,
            color,
            thickness,
        )


def render_image(session: ImageSession) -> np.ndarray:
    image = decode_image(session.image)
    for annotation in session.annotations:
        render_annotation(image, annotation)
    return image


def resize_for_export(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    longest = max(width, height)
    if longest <= MAX_EXPORT_SIDE:
        return image
    scale = MAX_EXPORT_SIDE / longest
    size = (int(width * scale), int(height * scale))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def apply_filter(image: np.ndarray, filter_type: str) -> np.ndarray:
    if filter_type == "grayscale":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if filter_type == "blur":
        return cv2.GaussianBlur(image, (15, 15), 0)
    if filter_type == "sharpen":
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        return cv2.filter2D(image, -1, kernel)
    if filter_type == "edge_detection":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    if filter_type == "contrast":
        return cv2.convertScaleAbs(image, alpha=1.5, beta=0)
    if filter_type == "color_thresholding":
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([10, 255, 255]))
        return cv2.bitwise_and(image, image, mask=mask)
    if filter_type == "laplacian":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return cv2.cvtColor(np.uint8(np.absolute(laplacian)), cv2.COLOR_GRAY2BGR)
    if filter_type == "thermal":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    if filter_type == "high_pass":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        high_pass = cv2.filter2D(gray, -1, kernel)
        return cv2.cvtColor(high_pass, cv2.COLOR_GRAY2BGR)
    raise HTTPException(status_code=400, detail="Unknown filter_type")


def class_name_for(model_names: object, class_id: int) -> str:
    if isinstance(model_names, dict):
        return str(model_names.get(class_id, class_id))
    if isinstance(model_names, list) and 0 <= class_id < len(model_names):
        return str(model_names[class_id])
    return str(class_id)


def apply_annotation_update(update: AnnotationUpdate) -> ImageSession:
    session = sessions.get(update.image_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Unknown image_id")

    if update.replace:
        session.annotations = update.annotations
    else:
        remove_ids = set(update.remove_ids)
        session.annotations = [annotation for annotation in session.annotations if annotation.id not in remove_ids]
        session.annotations.extend(update.annotations)

    return session


def detect_session(image_id: str, confidence_threshold: float | None = None, iou_threshold: float | None = None) -> DetectResponse:
    session = sessions.get(image_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Unknown image_id")

    image = decode_image(session.image)
    height, width = image.shape[:2]
    yolo_model = get_model()

    with tempfile.NamedTemporaryFile(suffix=Path(session.filename).suffix or ".jpg", delete=False) as tmp:
        tmp.write(session.image)
        tmp_path = tmp.name
    try:
        kwargs = {}
        if confidence_threshold is not None:
            kwargs["conf"] = confidence_threshold
        if iou_threshold is not None:
            kwargs["iou"] = iou_threshold
        results = yolo_model(tmp_path, **kwargs)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    detections: list[Annotation] = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            detections.append(
                Annotation(
                    type="rectangle",
                    start=Point(x=x1, y=y1),
                    end=Point(x=x2, y=y2),
                    color=(255, 0, 0),
                    thickness=2,
                    source="yolo",
                    class_name=class_name_for(yolo_model.names, class_id),
                    confidence=confidence,
                )
            )

    session.annotations = detections.copy()
    return DetectResponse(image_id=image_id, filename=session.filename, width=width, height=height, detections=detections)


def severity_from_box(annotation: Annotation, image_width: int, image_height: int) -> str:
    """Compute severity tier from bounding box pixel area * confidence.

    Returns 'minor', 'moderate', or 'severe'.
    For non-rectangle annotations or missing coordinates, returns 'minor' as safe default.
    """
    if annotation.type != "rectangle" or not annotation.start or not annotation.end:
        return "minor"

    pixel_area = (annotation.end.x - annotation.start.x) * (annotation.end.y - annotation.start.y)
    confidence = annotation.confidence if annotation.confidence is not None else 1.0
    score = pixel_area * confidence

    if score <= MINOR_MAX:
        return "minor"
    if score <= MODERATE_MAX:
        return "moderate"
    return "severe"


def _is_boundary_case(score: float) -> bool:
    """Return True if score is within 10% of MINOR_MAX or MODERATE_MAX."""
    minor_lo = MINOR_MAX * (1 - BOUNDARY_MARGIN)
    minor_hi = MINOR_MAX * (1 + BOUNDARY_MARGIN)
    moderate_lo = MODERATE_MAX * (1 - BOUNDARY_MARGIN)
    moderate_hi = MODERATE_MAX * (1 + BOUNDARY_MARGIN)
    return (minor_lo <= score <= minor_hi) or (moderate_lo <= score <= moderate_hi)


def build_ai_prompt(detections: list[Annotation], image_width: int, image_height: int) -> str:
    """Build the text prompt for the AI provider."""
    valid_classes_list = sorted(VALID_CLASSES)

    detection_entries = []
    for i, ann in enumerate(detections):
        if ann.type == "rectangle" and ann.start and ann.end:
            pixel_area = (ann.end.x - ann.start.x) * (ann.end.y - ann.start.y)
            confidence = ann.confidence if ann.confidence is not None else 1.0
            score = pixel_area * confidence
            is_boundary = _is_boundary_case(score)
            formula_severity = severity_from_box(ann, image_width, image_height)
            entry = {
                "index": i,
                "class_name": ann.class_name,
                "confidence": round(ann.confidence, 3) if ann.confidence is not None else None,
                "bbox": {"x1": ann.start.x, "y1": ann.start.y, "x2": ann.end.x, "y2": ann.end.y},
                "formula_severity": formula_severity,
                "is_boundary_case": is_boundary,
            }
        else:
            entry = {
                "index": i,
                "class_name": ann.class_name,
                "confidence": ann.confidence,
                "formula_severity": "minor",
                "is_boundary_case": False,
            }
        detection_entries.append(entry)

    prompt = f"""You are a car damage assessment AI. You are given a car damage image along with YOLO detections.

Image dimensions: {image_width}x{image_height} pixels

YOLO Detections:
{json.dumps(detection_entries, indent=2)}

Valid class names (you MUST use only these exact strings):
{json.dumps(valid_classes_list)}

Your tasks:
1. Review each detection. Confirm or override its severity (minor/moderate/severe). Pay special attention to detections marked "is_boundary_case": true — use your visual judgment for those.
2. Identify any likely missed damage regions not in the detection list. Return them with bounding box coordinates and a class_name from the valid list.
3. Flag any detection index that appears to be a false positive (e.g., non-damage area incorrectly detected).

Respond ONLY with a JSON object matching this exact schema (no markdown, no explanation):
{{
  "enriched": [
    {{"index": 0, "severity": "moderate", "class_name": "damaged-door", "confidence": 0.82}}
  ],
  "missed": [
    {{"class_name": "damaged-fender", "x1": 100, "y1": 200, "x2": 300, "y2": 400, "severity": "minor"}}
  ],
  "false_positive_indices": [2]
}}

Rules:
- Every index in "enriched" must correspond to a valid detection index (0 to {len(detections) - 1})
- All class_name values must be from the valid list above
- Bounding box coordinates must be within image bounds (0 to {image_width} for x, 0 to {image_height} for y)
- If no missed detections, use empty array for "missed"
- If no false positives, use empty array for "false_positive_indices"
- Return only the JSON object, nothing else
"""
    return prompt


def call_claude(prompt: str, image_bytes: bytes, model_name: str, api_key: str) -> str:
    """Call Anthropic Claude API with vision. Returns raw JSON string from AI response."""
    try:
        import anthropic  # type: ignore

        client = anthropic.Anthropic(api_key=api_key)
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        message = client.messages.create(
            model=model_name,
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_b64,
                            },
                        },
                        {
                            "type": "text",
                            "text": prompt,
                        },
                    ],
                }
            ],
        )
        return message.content[0].text
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"claude API error: {e}")


def call_gemini(prompt: str, image_bytes: bytes, model_name: str, api_key: str) -> str:
    """Call Google Gemini API with vision. Returns raw JSON string from AI response."""
    try:
        import google.generativeai as genai  # type: ignore

        genai.configure(api_key=api_key)
        generation_config = genai.GenerationConfig(max_output_tokens=1024)
        gemini_model = genai.GenerativeModel(model_name=model_name, generation_config=generation_config)
        image_part = {"mime_type": "image/jpeg", "data": image_bytes}
        response = gemini_model.generate_content([image_part, prompt])
        return response.text
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"gemini API error: {e}")


def call_openai(prompt: str, image_bytes: bytes, model_name: str, api_key: str) -> str:
    """Call OpenAI API with vision. Returns raw JSON string from AI response."""
    try:
        from openai import OpenAI  # type: ignore

        client = OpenAI(api_key=api_key)
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        data_url = f"data:image/jpeg;base64,{image_b64}"
        response = client.chat.completions.create(
            model=model_name,
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": data_url},
                        },
                        {
                            "type": "text",
                            "text": prompt,
                        },
                    ],
                }
            ],
        )
        return response.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"openai API error: {e}")


@app.get("/", response_class=HTMLResponse)
def root() -> Response:
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return HTMLResponse("<!doctype html><title>Car Damage Annotation Tool</title><h1>Backend is running</h1>")


@app.post("/images", response_model=list[ImageUploadResponse])
async def upload_images(files: list[UploadFile] = File(...)) -> list[ImageUploadResponse]:
    uploaded: list[ImageUploadResponse] = []
    for file in files:
        image_bytes = await file.read()
        image = decode_image(image_bytes)
        height, width = image.shape[:2]
        image_id = str(uuid.uuid4())
        filename = file.filename or f"{image_id}.jpg"
        sessions[image_id] = ImageSession(filename=filename, image=image_bytes, original_image=image_bytes, annotations=[])
        uploaded.append(ImageUploadResponse(image_id=image_id, filename=filename, width=width, height=height))
    return uploaded


@app.post("/detect", response_model=DetectResponse)
async def detect(
    file: UploadFile | None = File(None),
    image_id: str | None = Form(None),
    confidence_threshold: float | None = Form(None),
    iou_threshold: float | None = Form(None),
) -> DetectResponse:
    if file is not None:
        image_bytes = await file.read()
        image = decode_image(image_bytes)
        height, width = image.shape[:2]
        image_id = str(uuid.uuid4())
        sessions[image_id] = ImageSession(filename=file.filename or f"{image_id}.jpg", image=image_bytes, original_image=image_bytes, annotations=[])
        response = detect_session(image_id, confidence_threshold, iou_threshold)
        response.width = width
        response.height = height
        return response

    if image_id:
        return detect_session(image_id, confidence_threshold, iou_threshold)

    raise HTTPException(status_code=400, detail="Provide either file or image_id")


@app.post("/annotations")
def update_annotations(update: AnnotationUpdate) -> StreamingResponse:
    session = apply_annotation_update(update)
    rendered = render_image(session)
    return StreamingResponse(io.BytesIO(encode_jpeg(rendered)), media_type="image/jpeg")


@app.post("/annotations/render")
def render_annotations(update: AnnotationUpdate) -> StreamingResponse:
    return update_annotations(update)


@app.post("/images/edit")
def edit_image(edit: ImageEditRequest) -> StreamingResponse:
    session = sessions.get(edit.image_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Unknown image_id")

    if edit.action == "reset":
        session.image = session.original_image
        return StreamingResponse(io.BytesIO(session.image), media_type="image/jpeg")

    image = decode_image(session.image)
    if edit.action == "rotate":
        if edit.degrees == 90:
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif edit.degrees == 180:
            image = cv2.rotate(image, cv2.ROTATE_180)
        elif edit.degrees == 270:
            image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            raise HTTPException(status_code=400, detail="degrees must be 90, 180, or 270")
    elif edit.action == "flip":
        if edit.axis == "horizontal":
            image = cv2.flip(image, 1)
        elif edit.axis == "vertical":
            image = cv2.flip(image, 0)
        else:
            raise HTTPException(status_code=400, detail="axis must be horizontal or vertical")
    elif edit.action == "crop":
        if not edit.start or not edit.end:
            raise HTTPException(status_code=400, detail="crop requires start and end")
        height, width = image.shape[:2]
        start_x = min(width, max(0, edit.start.x))
        end_x = min(width, max(0, edit.end.x))
        start_y = min(height, max(0, edit.start.y))
        end_y = min(height, max(0, edit.end.y))
        x1, x2 = sorted((start_x, end_x))
        y1, y2 = sorted((start_y, end_y))
        if x2 <= x1 or y2 <= y1:
            raise HTTPException(status_code=400, detail="crop area is empty")
        image = image[y1:y2, x1:x2]
    elif edit.action == "filter":
        if not edit.filter_type:
            raise HTTPException(status_code=400, detail="filter_type is required")
        image = apply_filter(image, edit.filter_type)

    session.image = encode_like_upload(image, session.filename)
    return StreamingResponse(io.BytesIO(session.image), media_type="image/jpeg")


@app.post("/ai-analysis", response_model=DetectResponse)
async def ai_analysis(
    request: AIAnalysisRequest,
    x_ai_api_key: str | None = Header(default=None, alias="X-AI-Api-Key"),
) -> DetectResponse:
    """Run YOLO detection then enrich with AI provider (severity, missed detections, FP flagging)."""
    if request.image_id not in sessions:
        raise HTTPException(status_code=404, detail="Unknown image_id")

    if not x_ai_api_key:
        raise HTTPException(status_code=401, detail="X-AI-Api-Key header is required")

    session = sessions[request.image_id]

    # Run YOLO detection to get fresh detections
    detect_result = detect_session(request.image_id)
    detections = detect_result.detections

    # Decode image to get dimensions
    image_np = decode_image(session.image)
    image_height, image_width = image_np.shape[:2]

    # Assign formula-based severity to each detection
    for ann in detections:
        ann.severity = severity_from_box(ann, image_width, image_height)

    # If no detections, return immediately with formula-only results
    if not detections:
        session.annotations = detections
        return DetectResponse(
            image_id=request.image_id,
            filename=session.filename,
            width=image_width,
            height=image_height,
            detections=detections,
        )

    # Build AI prompt
    prompt = build_ai_prompt(detections, image_width, image_height)

    # Encode image as JPEG bytes for AI provider
    image_bytes = encode_jpeg(image_np)

    # Dispatch to appropriate provider
    ai_raw: str = ""
    try:
        if request.provider == "claude":
            ai_raw = call_claude(prompt, image_bytes, request.model, x_ai_api_key)
        elif request.provider == "gemini":
            ai_raw = call_gemini(prompt, image_bytes, request.model, x_ai_api_key)
        elif request.provider == "openai":
            ai_raw = call_openai(prompt, image_bytes, request.model, x_ai_api_key)
    except HTTPException:
        # Provider call failed — return formula-only results (graceful degradation)
        session.annotations = detections
        return DetectResponse(
            image_id=request.image_id,
            filename=session.filename,
            width=image_width,
            height=image_height,
            detections=detections,
        )

    # Parse AI JSON response; degrade gracefully on parse failure
    ai_data: dict = {}
    try:
        # Strip potential markdown code fences
        raw = ai_raw.strip()
        if raw.startswith("```"):
            lines = raw.split("\n")
            raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        ai_data = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        # Cannot parse AI response — return formula-only results
        session.annotations = detections
        return DetectResponse(
            image_id=request.image_id,
            filename=session.filename,
            width=image_width,
            height=image_height,
            detections=detections,
        )

    # Apply AI enrichment to existing detections
    enriched_list = ai_data.get("enriched", [])
    for item in enriched_list:
        idx = item.get("index")
        if idx is None or not isinstance(idx, int) or idx < 0 or idx >= len(detections):
            continue
        ann = detections[idx]
        if "severity" in item and item["severity"] in ("minor", "moderate", "severe"):
            ann.severity = item["severity"]
        if "class_name" in item and item["class_name"] in VALID_CLASSES:
            ann.class_name = item["class_name"]
        if "confidence" in item and isinstance(item["confidence"], (int, float)):
            ann.confidence = float(item["confidence"])
        ann.ai_source = request.provider

    # Add missed detections
    missed_list = ai_data.get("missed", [])
    for item in missed_list:
        class_name = item.get("class_name")
        if class_name not in VALID_CLASSES:
            continue
        x1 = int(item.get("x1", 0))
        y1 = int(item.get("y1", 0))
        x2 = int(item.get("x2", 0))
        y2 = int(item.get("y2", 0))
        # Clip coordinates to image dimensions
        x1 = max(0, min(x1, image_width))
        y1 = max(0, min(y1, image_height))
        x2 = max(0, min(x2, image_width))
        y2 = max(0, min(y2, image_height))
        if x2 <= x1 or y2 <= y1:
            continue
        severity = item.get("severity", "minor")
        if severity not in ("minor", "moderate", "severe"):
            severity = "minor"
        new_ann = Annotation(
            type="rectangle",
            start=Point(x=x1, y=y1),
            end=Point(x=x2, y=y2),
            color=(255, 165, 0),
            thickness=2,
            source="ai-missed",
            ai_source=request.provider,
            class_name=class_name,
            confidence=item.get("confidence", 0.5),
            severity=severity,
        )
        detections.append(new_ann)

    # Flag false positives
    fp_indices = ai_data.get("false_positive_indices", [])
    for idx in fp_indices:
        if isinstance(idx, int) and 0 <= idx < len(detections):
            detections[idx].source = "ai-fp"

    # Update session annotations with enriched list
    session.annotations = detections

    return DetectResponse(
        image_id=request.image_id,
        filename=session.filename,
        width=image_width,
        height=image_height,
        detections=detections,
    )


@app.get("/export")
def export_zip(zip_name: str = "annotated_images", max_size_mb: float | None = None) -> StreamingResponse:
    if not sessions:
        raise HTTPException(status_code=404, detail="No annotated images in session")

    zip_buffer = io.BytesIO()
    max_size_bytes = int(max_size_mb * 1024 * 1024) if max_size_mb else None
    current_size = 0
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for image_id, session in sessions.items():
            rendered = resize_for_export(render_image(session))
            filename = Path(session.filename).stem or image_id
            data = encode_jpeg(rendered)
            if max_size_bytes and current_size + len(data) > max_size_bytes:
                continue
            archive.writestr(f"{filename}_annotated.jpg", data)
            current_size += len(data)
            # Write per-image metadata JSON sidecar with severity information
            metadata = {
                "image_id": image_id,
                "filename": session.filename,
                "annotations": [
                    {
                        "id": ann.id,
                        "class_name": ann.class_name,
                        "confidence": ann.confidence,
                        "severity": ann.severity,
                        "source": ann.source,
                        "ai_source": ann.ai_source,
                    }
                    for ann in session.annotations
                ],
            }
            archive.writestr(f"{filename}_metadata.json", json.dumps(metadata, indent=2))
    zip_buffer.seek(0)
    safe_name = "".join(char for char in zip_name if char.isalnum() or char in ("-", "_")).strip() or "annotated_images"
    headers = {"Content-Disposition": f'attachment; filename="{safe_name}.zip"'}
    return StreamingResponse(zip_buffer, media_type="application/zip", headers=headers)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model_path": str(MODEL_PATH)}

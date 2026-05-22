from __future__ import annotations

import io
import os
import tempfile
import uuid
import zipfile
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"
MODEL_PATH = Path(os.getenv("YOLO_MODEL_PATH", BASE_DIR / "best.pt"))
MAX_EXPORT_SIDE = 1024
JPEG_QUALITY = 85

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
    zip_buffer.seek(0)
    safe_name = "".join(char for char in zip_name if char.isalnum() or char in ("-", "_")).strip() or "annotated_images"
    headers = {"Content-Disposition": f'attachment; filename="{safe_name}.zip"'}
    return StreamingResponse(zip_buffer, media_type="application/zip", headers=headers)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model_path": str(MODEL_PATH)}

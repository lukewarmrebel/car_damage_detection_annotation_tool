const canvas = document.querySelector("#canvas");
const ctx = canvas.getContext("2d");
const fileInput = document.querySelector("#fileInput");
const imageList = document.querySelector("#imageList");
const imageCounter = document.querySelector("#imageCounter");
const statusText = document.querySelector("#statusText");
const emptyState = document.querySelector("#emptyState");
const detectBtn = document.querySelector("#detectBtn");
const exportBtn = document.querySelector("#exportBtn");
const prevBtn = document.querySelector("#prevBtn");
const nextBtn = document.querySelector("#nextBtn");
const undoBtn = document.querySelector("#undoBtn");
const clearBtn = document.querySelector("#clearBtn");
const resetViewBtn = document.querySelector("#resetViewBtn");
const helpBtn = document.querySelector("#helpBtn");
const closeHelpBtn = document.querySelector("#closeHelpBtn");
const helpDialog = document.querySelector("#helpDialog");
const compareBtn = document.querySelector("#compareBtn");
const compareDialog = document.querySelector("#compareDialog");
const closeCompareBtn = document.querySelector("#closeCompareBtn");
const compareA = document.querySelector("#compareA");
const compareB = document.querySelector("#compareB");
const compareImageA = document.querySelector("#compareImageA");
const compareImageB = document.querySelector("#compareImageB");
const colorInput = document.querySelector("#colorInput");
const thicknessInput = document.querySelector("#thicknessInput");
const fontSizeInput = document.querySelector("#fontSizeInput");
const confidenceInput = document.querySelector("#confidenceInput");
const iouInput = document.querySelector("#iouInput");
const zipNameInput = document.querySelector("#zipNameInput");
const zipSizeInput = document.querySelector("#zipSizeInput");
const aiAnalyzeBtn = document.querySelector("#aiAnalyzeBtn");
const autoAnalyzeBtn = document.querySelector("#autoAnalyzeBtn");
const settingsBtn = document.querySelector("#settingsBtn");
const settingsDialog = document.querySelector("#settingsDialog");
const aiProviderSelect = document.querySelector("#aiProviderSelect");
const aiModelSelect = document.querySelector("#aiModelSelect");
const aiApiKeyInput = document.querySelector("#aiApiKeyInput");
const saveSettingsBtn = document.querySelector("#saveSettingsBtn");
const closeSettingsBtn = document.querySelector("#closeSettingsBtn");
const sessionSummary = document.querySelector("#sessionSummary");

const state = {
  images: [],
  current: -1,
  tool: "select",
  scale: 1,
  offsetX: 0,
  offsetY: 0,
  drawing: false,
  panning: false,
  dragStart: null,
  preview: null,
  yoloProcessed: new Set(),
  aiProcessed: new Set(),
};

function activeImage() {
  return state.images[state.current] || null;
}

function setStatus(message) {
  statusText.textContent = message;
}

function makeId() {
  return crypto.randomUUID ? crypto.randomUUID() : `${Date.now()}-${Math.random()}`;
}

function hexToRgb(hex) {
  const value = hex.replace("#", "");
  return [
    parseInt(value.slice(0, 2), 16),
    parseInt(value.slice(2, 4), 16),
    parseInt(value.slice(4, 6), 16),
  ];
}

function rgbToCss(color) {
  return `rgb(${color[0]}, ${color[1]}, ${color[2]})`;
}

function severityColor(severity) {
  if (severity === "minor") return [255, 215, 0];
  if (severity === "moderate") return [255, 140, 0];
  if (severity === "severe") return [255, 51, 51];
  return null;
}

function resizeCanvas() {
  const rect = canvas.parentElement.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.floor(rect.width * ratio));
  canvas.height = Math.max(1, Math.floor(rect.height * ratio));
  canvas.style.width = `${rect.width}px`;
  canvas.style.height = `${rect.height}px`;
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  draw();
}

function fitImage() {
  const image = activeImage();
  if (!image || !image.element) return;
  const rect = canvas.getBoundingClientRect();
  const fit = Math.min(rect.width / image.width, rect.height / image.height) * 0.92;
  state.scale = Number.isFinite(fit) && fit > 0 ? fit : 1;
  state.offsetX = (rect.width - image.width * state.scale) / 2;
  state.offsetY = (rect.height - image.height * state.scale) / 2;
}

function screenToImage(event) {
  const rect = canvas.getBoundingClientRect();
  return {
    x: Math.round((event.clientX - rect.left - state.offsetX) / state.scale),
    y: Math.round((event.clientY - rect.top - state.offsetY) / state.scale),
    sx: event.clientX - rect.left,
    sy: event.clientY - rect.top,
  };
}

function drawAnnotation(annotation) {
  ctx.save();
  ctx.strokeStyle = rgbToCss(annotation.color || [255, 0, 0]);
  ctx.fillStyle = ctx.strokeStyle;
  ctx.lineWidth = (annotation.thickness || 2) / state.scale;
  ctx.font = `${Math.max(12 / state.scale, 7)}px Arial`;

  if (annotation.type === "rectangle" && annotation.start && annotation.end) {
    const x = annotation.start.x;
    const y = annotation.start.y;
    const w = annotation.end.x - annotation.start.x;
    const h = annotation.end.y - annotation.start.y;
    ctx.strokeRect(x, y, w, h);
    if (annotation.severity) {
      const sc = severityColor(annotation.severity);
      if (sc) ctx.strokeStyle = rgbToCss(sc);
      ctx.strokeRect(x, y, w, h);
    }
    if (annotation.class_name) {
      const fpTag = annotation.source === "ai-fp" ? " [?FP]" : "";
      const sevTag = annotation.severity ? ` [${annotation.severity}]` : "";
      const label = `${annotation.class_name}${sevTag}${fpTag} ${annotation.confidence ? Math.round(annotation.confidence * 100) + "%" : ""}`.trim();
      ctx.fillText(label, x, Math.max(y - 8 / state.scale, 12 / state.scale));
    }
  }

  if (annotation.type === "circle" && annotation.center && annotation.radius) {
    ctx.beginPath();
    ctx.arc(annotation.center.x, annotation.center.y, annotation.radius, 0, Math.PI * 2);
    ctx.stroke();
  }

  if (annotation.type === "text" && annotation.position && annotation.text) {
    ctx.font = `${annotation.font_size || Math.max((annotation.font_scale || 0.7) * 24, 12)}px Arial`;
    ctx.fillText(annotation.text, annotation.position.x, annotation.position.y);
  }

  ctx.restore();
}

function draw() {
  const rect = canvas.getBoundingClientRect();
  ctx.clearRect(0, 0, rect.width, rect.height);
  const image = activeImage();
  emptyState.style.display = image ? "none" : "grid";
  if (!image || !image.element) return;

  ctx.save();
  ctx.translate(state.offsetX, state.offsetY);
  ctx.scale(state.scale, state.scale);
  ctx.drawImage(image.element, 0, 0, image.width, image.height);
  image.annotations.forEach(drawAnnotation);
  if (state.preview) drawAnnotation(state.preview);
  ctx.restore();
}

function refreshImageList() {
  imageList.innerHTML = "";
  state.images.forEach((image, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `image-item${index === state.current ? " active" : ""}`;
    button.innerHTML = `<span class="image-name"></span><span class="badge"></span>`;
    button.querySelector(".image-name").textContent = image.filename;
    button.querySelector(".badge").textContent = image.annotations.length ? image.annotations.length : "";
    button.addEventListener("click", () => selectImage(index));
    imageList.appendChild(button);
  });
  imageCounter.textContent = state.images.length ? `${state.current + 1} / ${state.images.length}` : "0 / 0";
}

function selectImage(index) {
  if (index < 0 || index >= state.images.length) return;
  state.current = index;
  state.preview = null;
  fitImage();
  refreshImageList();
  draw();
  setStatus(`Viewing ${activeImage().filename}`);
}

async function loadImageElement(file) {
  const url = URL.createObjectURL(file);
  const element = new Image();
  element.src = url;
  await element.decode();
  return { element, url };
}

async function uploadFiles(files) {
  if (!files.length) return;
  setStatus("Uploading images...");
  const form = new FormData();
  [...files].forEach((file) => form.append("files", file));
  const response = await fetch("/images", { method: "POST", body: form });
  if (!response.ok) throw new Error(await response.text());
  const uploaded = await response.json();

  for (const [index, file] of [...files].entries()) {
    const meta = uploaded[index];
    const loaded = await loadImageElement(file);
    state.images.push({
      image_id: meta.image_id,
      filename: meta.filename,
      width: meta.width,
      height: meta.height,
      annotations: [],
      dirty: false,
      saving: null,
      element: loaded.element,
      objectUrl: loaded.url,
    });
  }

  if (state.current === -1) selectImage(0);
  refreshImageList();
  setStatus(`Loaded ${files.length} image${files.length === 1 ? "" : "s"}.`);
}

async function runDetection() {
  const image = activeImage();
  if (!image) return;
  if (state.yoloProcessed.has(image.image_id) && !confirm("Detection already ran for this image. Run it again and replace current annotations?")) {
    return;
  }
  detectBtn.disabled = true;
  setStatus("Running damage detection...");
  const form = new FormData();
  form.append("image_id", image.image_id);
  form.append("confidence_threshold", confidenceInput.value);
  form.append("iou_threshold", iouInput.value);
  const response = await fetch("/detect", { method: "POST", body: form });
  detectBtn.disabled = false;
  if (!response.ok) {
    setStatus("Detection failed.");
    throw new Error(await response.text());
  }
  const result = await response.json();
  image.annotations = result.detections;
  image.dirty = false;
  state.yoloProcessed.add(image.image_id);
  refreshImageList();
  draw();
  setStatus(`Detection complete: ${result.detections.length} box${result.detections.length === 1 ? "" : "es"}.`);
}

async function replaceCurrentImageFromResponse(response, options = {}) {
  if (!response.ok) throw new Error(await response.text());
  const image = activeImage();
  const keepAnnotations = options.keepAnnotations !== false;
  const existingAnnotations = keepAnnotations ? image.annotations : [];
  const blob = await response.blob();
  if (image.objectUrl) URL.revokeObjectURL(image.objectUrl);
  image.objectUrl = URL.createObjectURL(blob);
  image.element = new Image();
  image.element.src = image.objectUrl;
  await image.element.decode();
  image.width = image.element.naturalWidth;
  image.height = image.element.naturalHeight;
  image.annotations = existingAnnotations;
  image.dirty = keepAnnotations;
  state.yoloProcessed.delete(image.image_id);
  fitImage();
  refreshImageList();
  draw();
  if (image.dirty) await syncImage(image, { silent: true });
}

async function editCurrentImage(payload) {
  const image = activeImage();
  if (!image) return;
  setStatus("Updating image...");
  const response = await fetch("/images/edit", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image_id: image.image_id, ...payload }),
  });
  await replaceCurrentImageFromResponse(response);
  setStatus(`Updated ${image.filename}`);
}

function annotationFromDrag(start, end) {
  const base = {
    id: makeId(),
    color: hexToRgb(colorInput.value),
    thickness: Number(thicknessInput.value),
  };
  if (state.tool === "rectangle" || state.tool === "crop") {
    return { ...base, type: "rectangle", start: { x: start.x, y: start.y }, end: { x: end.x, y: end.y } };
  }
  const radius = Math.round(Math.hypot(end.x - start.x, end.y - start.y));
  return { ...base, type: "circle", center: { x: start.x, y: start.y }, radius };
}

function hitTest(annotation, point) {
  if (annotation.type === "rectangle" && annotation.start && annotation.end) {
    const minX = Math.min(annotation.start.x, annotation.end.x);
    const maxX = Math.max(annotation.start.x, annotation.end.x);
    const minY = Math.min(annotation.start.y, annotation.end.y);
    const maxY = Math.max(annotation.start.y, annotation.end.y);
    return point.x >= minX && point.x <= maxX && point.y >= minY && point.y <= maxY;
  }
  if (annotation.type === "circle" && annotation.center) {
    return Math.hypot(point.x - annotation.center.x, point.y - annotation.center.y) <= annotation.radius;
  }
  if (annotation.type === "text" && annotation.position) {
    return Math.abs(point.x - annotation.position.x) < 80 && Math.abs(point.y - annotation.position.y) < 28;
  }
  return false;
}

async function syncCurrentImage(options = {}) {
  const image = activeImage();
  if (!image) return false;
  return syncImage(image, options);
}

async function syncImage(image, options = {}) {
  if (!image) return false;
  image.saving = fetch("/annotations", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image_id: image.image_id, annotations: image.annotations, replace: true }),
  });
  const response = await image.saving;
  image.saving = null;
  if (!response.ok) {
    image.dirty = true;
    setStatus(`Could not save annotations for ${image.filename}.`);
    throw new Error(await response.text());
  }
  image.dirty = false;
  if (!options.silent) setStatus(`Saved annotations for ${image.filename}.`);
  return true;
}

async function syncAllImages() {
  for (const image of state.images) {
    if (image.saving) await image.saving;
    if (image.dirty) await syncImage(image, { silent: true });
  }
}

canvas.addEventListener("pointerdown", async (event) => {
  const image = activeImage();
  if (!image) return;
  const point = screenToImage(event);

  if (event.button === 2) {
    const index = image.annotations.findLastIndex((annotation) => hitTest(annotation, point));
    if (index >= 0) {
      image.annotations.splice(index, 1);
      image.dirty = true;
      await syncCurrentImage();
      refreshImageList();
      draw();
    }
    return;
  }

  if (state.tool === "select" || event.button === 1 || event.shiftKey) {
    state.panning = true;
    state.dragStart = { sx: point.sx, sy: point.sy, offsetX: state.offsetX, offsetY: state.offsetY };
    canvas.setPointerCapture(event.pointerId);
    return;
  }

  if (state.tool === "erase") {
    const index = image.annotations.findLastIndex((annotation) => hitTest(annotation, point));
    if (index >= 0) {
      image.annotations.splice(index, 1);
      image.dirty = true;
      await syncCurrentImage();
      refreshImageList();
      draw();
    }
    return;
  }

  if (state.tool === "text") {
    const text = prompt("Text annotation");
    if (text) {
      image.annotations.push({
        id: makeId(),
        type: "text",
        text,
        position: { x: point.x, y: point.y },
        color: hexToRgb(colorInput.value),
        thickness: Number(thicknessInput.value),
        font_scale: 0.7,
        font_size: Number(fontSizeInput.value),
      });
      image.dirty = true;
      await syncCurrentImage();
      refreshImageList();
      draw();
    }
    return;
  }

  state.drawing = true;
  state.dragStart = point;
  canvas.setPointerCapture(event.pointerId);
});

canvas.addEventListener("pointermove", (event) => {
  const point = screenToImage(event);
  if (state.panning && state.dragStart) {
    state.offsetX = state.dragStart.offsetX + point.sx - state.dragStart.sx;
    state.offsetY = state.dragStart.offsetY + point.sy - state.dragStart.sy;
    draw();
    return;
  }
  if (state.drawing && state.dragStart) {
    state.preview = annotationFromDrag(state.dragStart, point);
    draw();
  }
});

canvas.addEventListener("pointerup", async (event) => {
  const image = activeImage();
  const point = screenToImage(event);
  if (state.panning) {
    state.panning = false;
    state.dragStart = null;
    return;
  }
  if (state.drawing && image) {
    if (state.tool === "crop") {
      const crop = annotationFromDrag(state.dragStart, point);
      state.drawing = false;
      state.preview = null;
      state.dragStart = null;
      await editCurrentImage({ action: "crop", start: crop.start, end: crop.end });
      return;
    }
    const annotation = annotationFromDrag(state.dragStart, point);
    state.drawing = false;
    state.preview = null;
    state.dragStart = null;
    image.annotations.push(annotation);
    image.dirty = true;
    await syncCurrentImage();
    refreshImageList();
    draw();
  }
});

canvas.addEventListener("contextmenu", (event) => event.preventDefault());

canvas.addEventListener("wheel", (event) => {
  const image = activeImage();
  if (!image) return;
  event.preventDefault();
  const rect = canvas.getBoundingClientRect();
  const mouseX = event.clientX - rect.left;
  const mouseY = event.clientY - rect.top;
  const beforeX = (mouseX - state.offsetX) / state.scale;
  const beforeY = (mouseY - state.offsetY) / state.scale;
  const factor = event.deltaY < 0 ? 1.1 : 0.9;
  state.scale = Math.min(8, Math.max(0.08, state.scale * factor));
  state.offsetX = mouseX - beforeX * state.scale;
  state.offsetY = mouseY - beforeY * state.scale;
  draw();
}, { passive: false });

document.querySelectorAll(".tool-btn").forEach((button) => {
  button.addEventListener("click", () => {
    state.tool = button.dataset.tool;
    document.querySelectorAll(".tool-btn").forEach((item) => item.classList.toggle("active", item === button));
    canvas.style.cursor = state.tool === "select" ? "grab" : state.tool === "erase" ? "not-allowed" : "crosshair";
  });
});

fileInput.addEventListener("change", async (event) => {
  try {
    await uploadFiles(event.target.files);
  } catch (error) {
    console.error(error);
    setStatus("Upload failed.");
  } finally {
    fileInput.value = "";
  }
});

detectBtn.addEventListener("click", () => runDetection().catch((error) => console.error(error)));
prevBtn.addEventListener("click", () => selectImage(state.current - 1));
nextBtn.addEventListener("click", () => selectImage(state.current + 1));
resetViewBtn.addEventListener("click", () => {
  editCurrentImage({ action: "reset" }).catch((error) => {
    console.error(error);
    fitImage();
    draw();
  });
});

undoBtn.addEventListener("click", async () => {
  const image = activeImage();
  if (!image || !image.annotations.length) return;
  image.annotations.pop();
  image.dirty = true;
  await syncCurrentImage();
  refreshImageList();
  draw();
});

clearBtn.addEventListener("click", async () => {
  const image = activeImage();
  if (!image) return;
  image.annotations = [];
  state.yoloProcessed.delete(image.image_id);
  image.dirty = true;
  await syncCurrentImage();
  refreshImageList();
  draw();
});

exportBtn.addEventListener("click", async () => {
  try {
    setStatus("Saving annotations before export...");
    await syncAllImages();
    const params = new URLSearchParams();
    params.set("zip_name", zipNameInput.value || "annotated_images");
    if (zipSizeInput.value) params.set("max_size_mb", zipSizeInput.value);
    window.location.href = `/export?${params.toString()}`;
  } catch (error) {
    console.error(error);
    setStatus("Export stopped because annotations could not be saved.");
  }
});

document.querySelectorAll("[data-edit]").forEach((button) => {
  button.addEventListener("click", () => {
    const action = button.dataset.edit;
    if (action === "rotate") editCurrentImage({ action, degrees: Number(button.dataset.degrees) }).catch(console.error);
    if (action === "flip") editCurrentImage({ action, axis: button.dataset.axis }).catch(console.error);
  });
});

document.querySelectorAll("[data-filter]").forEach((button) => {
  button.addEventListener("click", () => editCurrentImage({ action: "filter", filter_type: button.dataset.filter }).catch(console.error));
});

function refreshCompareImages() {
  const first = state.images[Number(compareA.value)];
  const second = state.images[Number(compareB.value)];
  compareImageA.src = first ? first.objectUrl : "";
  compareImageB.src = second ? second.objectUrl : "";
}

compareBtn.addEventListener("click", () => {
  if (!state.images.length) return;
  compareA.innerHTML = "";
  compareB.innerHTML = "";
  state.images.forEach((image, index) => {
    compareA.add(new Option(`${index + 1}. ${image.filename}`, index));
    compareB.add(new Option(`${index + 1}. ${image.filename}`, index));
  });
  compareA.value = String(Math.max(0, state.current));
  compareB.value = String(Math.min(state.images.length - 1, Math.max(1, state.current + 1)));
  refreshCompareImages();
  compareDialog.showModal();
});
compareA.addEventListener("change", refreshCompareImages);
compareB.addEventListener("change", refreshCompareImages);
closeCompareBtn.addEventListener("click", () => compareDialog.close());
helpBtn.addEventListener("click", () => helpDialog.showModal());
closeHelpBtn.addEventListener("click", () => helpDialog.close());

document.addEventListener("keydown", (event) => {
  if (event.target.matches("input, textarea, select")) return;
  if (event.ctrlKey && event.key.toLowerCase() === "z") {
    event.preventDefault();
    undoBtn.click();
  } else if (event.ctrlKey && event.key.toLowerCase() === "o") {
    event.preventDefault();
    fileInput.click();
  } else if (event.ctrlKey && event.key.toLowerCase() === "s") {
    event.preventDefault();
    exportBtn.click();
  } else if (event.key === "Delete") {
    clearBtn.click();
  } else if (event.key === "ArrowLeft") {
    selectImage(state.current - 1);
  } else if (event.key === "ArrowRight") {
    selectImage(state.current + 1);
  } else if (event.key.toLowerCase() === "m") {
    document.querySelector('[data-tool="select"]')?.click();
  } else if (event.key.toLowerCase() === "d") {
    detectBtn.click();
  } else if (["1", "2", "3", "4", "5"].includes(event.key)) {
    const tools = ["rectangle", "circle", "text", "crop", "erase"];
    document.querySelector(`[data-tool="${tools[Number(event.key) - 1]}"]`)?.click();
  }
});

// --- AI Settings ---

function updateModelOptions() {
  const models = {
    claude: ["claude-haiku-4-5"],
    gemini: ["gemini-1.5-flash"],
    openai: ["gpt-4o-mini"],
  };
  aiModelSelect.innerHTML = (models[aiProviderSelect.value] || []).map(m => `<option value="${m}">${m}</option>`).join("");
  const saved = localStorage.getItem("ai_model_" + aiProviderSelect.value);
  if (saved) aiModelSelect.value = saved;
}

function loadSettings() {
  aiProviderSelect.value = localStorage.getItem("ai_provider") || "claude";
  aiApiKeyInput.value = localStorage.getItem("ai_api_key_" + aiProviderSelect.value) || "";
  updateModelOptions();
}

function saveSettings() {
  localStorage.setItem("ai_provider", aiProviderSelect.value);
  localStorage.setItem("ai_api_key_" + aiProviderSelect.value, aiApiKeyInput.value);
  localStorage.setItem("ai_model_" + aiProviderSelect.value, aiModelSelect.value);
  setStatus("AI settings saved.");
  settingsDialog.close();
}

aiProviderSelect.addEventListener("change", () => {
  updateModelOptions();
  aiApiKeyInput.value = localStorage.getItem("ai_api_key_" + aiProviderSelect.value) || "";
});

settingsBtn.addEventListener("click", () => { loadSettings(); settingsDialog.showModal(); });
saveSettingsBtn.addEventListener("click", saveSettings);
closeSettingsBtn.addEventListener("click", () => settingsDialog.close());

// --- Session Summary ---

function refreshSessionSummary() {
  if (!sessionSummary) return;
  const totals = { minor: 0, moderate: 0, severe: 0, unrated: 0 };
  let totalAnnotations = 0;
  state.images.forEach(img => {
    img.annotations.forEach(a => {
      totalAnnotations++;
      if (a.severity === "minor") totals.minor++;
      else if (a.severity === "moderate") totals.moderate++;
      else if (a.severity === "severe") totals.severe++;
      else totals.unrated++;
    });
  });
  if (totalAnnotations === 0) {
    sessionSummary.innerHTML = '<p class="muted-text">No annotations yet.</p>';
    return;
  }
  sessionSummary.innerHTML = `
    <div class="summary-row"><span>Total</span><span>${totalAnnotations}</span></div>
    <div class="summary-row"><span class="severity-minor">Minor</span><span>${totals.minor}</span></div>
    <div class="summary-row"><span class="severity-moderate">Moderate</span><span>${totals.moderate}</span></div>
    <div class="summary-row"><span class="severity-severe">Severe</span><span>${totals.severe}</span></div>
    <div class="summary-row"><span>Unrated</span><span>${totals.unrated}</span></div>
  `;
}

// --- AI Analyze (single image) ---

async function runAIAnalysis() {
  const image = activeImage();
  if (!image) return;
  const provider = localStorage.getItem("ai_provider") || "claude";
  const apiKey = localStorage.getItem("ai_api_key_" + provider) || "";
  if (!apiKey) { setStatus("No API key configured. Open Settings first."); return; }
  if (state.aiProcessed.has(image.image_id) && !confirm("AI analysis already ran for this image. Run it again?")) return;
  aiAnalyzeBtn.disabled = true;
  setStatus("Running AI analysis...");
  const model = aiModelSelect ? aiModelSelect.value : (provider === "claude" ? "claude-haiku-4-5" : provider === "gemini" ? "gemini-1.5-flash" : "gpt-4o-mini");
  const response = await fetch("/ai-analysis", {
    method: "POST",
    headers: { "Content-Type": "application/json", "X-AI-Api-Key": apiKey, "X-AI-Provider": provider, "X-AI-Model": model },
    body: JSON.stringify({ image_id: image.image_id, provider, model }),
  });
  aiAnalyzeBtn.disabled = false;
  if (!response.ok) { setStatus("AI analysis failed: " + (await response.text())); return; }
  const result = await response.json();
  image.annotations = result.detections;
  image.dirty = false;
  state.aiProcessed.add(image.image_id);
  refreshImageList();
  draw();
  refreshSessionSummary();
  setStatus(`AI analysis complete: ${result.detections.length} annotation(s).`);
}

aiAnalyzeBtn.addEventListener("click", () => runAIAnalysis().catch(e => { console.error(e); setStatus("AI analysis error."); }));

// --- Batch Auto-Analyze ---

function setBatchStatus(index, status) {
  const items = imageList.querySelectorAll(".image-item");
  if (!items[index]) return;
  const badge = items[index].querySelector(".badge");
  if (status === "spinner") badge.innerHTML = '<span class="batch-spinner"></span>';
  else if (status === "done") badge.innerHTML = '<span class="batch-done">&#10003;</span>';
  else if (status === "error") badge.innerHTML = '<span style="color:var(--danger)">!</span>';
}

async function runBatchAnalysis() {
  if (!state.images.length) return;
  const provider = localStorage.getItem("ai_provider") || "claude";
  const apiKey = localStorage.getItem("ai_api_key_" + provider) || "";
  if (!apiKey) { setStatus("No API key configured. Open Settings first."); return; }
  autoAnalyzeBtn.disabled = true;
  setStatus("Starting batch analysis...");
  for (let i = 0; i < state.images.length; i++) {
    const image = state.images[i];
    setBatchStatus(i, "spinner");
    const model = provider === "claude" ? "claude-haiku-4-5" : provider === "gemini" ? "gemini-1.5-flash" : "gpt-4o-mini";
    try {
      const response = await fetch("/ai-analysis", {
        method: "POST",
        headers: { "Content-Type": "application/json", "X-AI-Api-Key": apiKey },
        body: JSON.stringify({ image_id: image.image_id, provider, model }),
      });
      if (response.ok) {
        const result = await response.json();
        image.annotations = result.detections;
        image.dirty = false;
        state.aiProcessed.add(image.image_id);
        setBatchStatus(i, "done");
        if (i === state.current) { refreshImageList(); draw(); }
        else refreshImageList();
        refreshSessionSummary();
      } else {
        setBatchStatus(i, "error");
      }
    } catch (e) {
      setBatchStatus(i, "error");
    }
  }
  autoAnalyzeBtn.disabled = false;
  setStatus("Batch analysis complete.");
}

autoAnalyzeBtn.addEventListener("click", () => runBatchAnalysis().catch(e => { console.error(e); setStatus("Batch error."); }));

window.addEventListener("resize", resizeCanvas);
resizeCanvas();

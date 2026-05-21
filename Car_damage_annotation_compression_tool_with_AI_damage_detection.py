import tkinter as tk
from tkinter import filedialog, messagebox, colorchooser, simpledialog
import cv2
from PIL import Image, ImageTk
import numpy as np
import os
import sys
import zipfile
from ultralytics import YOLO

# Resource paths — work both from source and from a PyInstaller .exe
if getattr(sys, 'frozen', False):
    _BUNDLE_DIR = sys._MEIPASS          # bundled data (model weights)
    _BASE_DIR   = os.path.dirname(sys.executable)  # writable dir next to .exe
else:
    _BUNDLE_DIR = os.path.dirname(os.path.abspath(__file__))
    _BASE_DIR   = _BUNDLE_DIR

# Global Variables
image_paths = []
current_image_index = 0
output_dir = os.path.join(_BASE_DIR, "processed_images")
os.makedirs(output_dir, exist_ok=True)

# YOLOv8 Model
model = YOLO(os.path.join(_BUNDLE_DIR, 'best.pt'))
yolo_processed = set()  # Track images processed by YOLO

# Annotation settings
annotations_dict = {}  # Dictionary to store annotations for each image
current_color = (255, 0, 0)  # Default color: Red
current_tool = "circle"
is_drawing = False
start_x, start_y = None, None
thickness = 2  # Default thickness for shapes
font_size = 12  # Default font size for text

# Zoom & Pan Settings
zoom_level = 1.0
offset_x, offset_y = 0, 0

# Cropping settings
is_cropping = False
crop_start_x, crop_start_y = None, None
crop_end_x, crop_end_y = None, None

# Original image state
original_img_state = None  # Stores the original state of the image

# Compare View settings
compare_mode = False
compare_index_1 = None
compare_index_2 = None

# YOLO Thresholds
confidence_threshold = 0.5
iou_threshold = 0.5

# Live drawing preview
preview_shape = None

# Function to Load Bulk Images
def load_images():
    global image_paths, current_image_index, annotations_dict, original_img_state, yolo_processed
    files = filedialog.askopenfilenames(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    if not files:
        return

    image_paths = list(files)
    current_image_index = 0
    annotations_dict = {path: [] for path in image_paths}
    yolo_processed.clear()  # Reset processed images when new images are loaded
    load_current_image()
    update_image_counter()

# Function to Load the Current Image
def load_current_image():
    global image_paths, current_image_index, img, original_img, tk_img, zoom_level, offset_x, offset_y, original_img_state
    if not image_paths:
        return

    image_path = image_paths[current_image_index]
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    original_img = img.copy()
    original_img_state = original_img.copy()  # Save the original state of the image
    
    zoom_level = 1.0
    offset_x, offset_y = 0, 0
    update_canvas()

# Function to run YOLO detection
def run_yolo_detection():
    global original_img, annotations_dict
    if not image_paths:
        messagebox.showwarning("No Image", "Please load an image first.")
        return

    image_path = image_paths[current_image_index]
    if image_path in yolo_processed:
        messagebox.showinfo("Already Processed",
            "This image has already been processed by YOLO. Clear markings to re-run.")
        return

    root.config(cursor="wait")
    detect_btn.config(text="⏳  Detecting...", state="disabled")
    root.update_idletasks()

    try:
        yolo_img = cv2.imread(image_path)
        yolo_img = cv2.cvtColor(yolo_img, cv2.COLOR_BGR2RGB)
        results    = model.predict(yolo_img, conf=confidence_threshold, iou=iou_threshold)
        class_names = model.names

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                class_name = class_names[int(box.cls[0])]
                annotations_dict[image_path].append({
                    "type": "rectangle",
                    "start": (x1, y1), "end": (x2, y2),
                    "color": (0, 255, 0), "thickness": thickness,
                    "source": "yolo", "class_name": class_name
                })
        yolo_processed.add(image_path)
        update_canvas()
    finally:
        root.config(cursor="")
        detect_btn.config(text="⚡  Detect Damage", state="normal")


def _render_shape(img_copy, shape):
    if shape["type"] == "circle":
        cv2.circle(img_copy, shape["center"], shape["radius"], shape["color"], shape["thickness"])
    elif shape["type"] == "rectangle":
        cv2.rectangle(img_copy, shape["start"], shape["end"], shape["color"], shape["thickness"])
        if shape.get("source") == "yolo" and shape.get("class_name"):
            tx, ty = shape["start"][0], shape["start"][1] - 10
            if ty < 0: ty = 10
            cv2.putText(img_copy, shape["class_name"], (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, shape["color"], 2)
    elif shape["type"] == "text":
        cv2.putText(img_copy, shape["text"], shape["position"],
                    cv2.FONT_HERSHEY_SIMPLEX, shape["font_scale"], shape["color"], shape["thickness"])

def update_canvas():
    global tk_img, img
    img_copy = original_img.copy()
    current_image_path = image_paths[current_image_index]
    for shape in annotations_dict[current_image_path]:
        _render_shape(img_copy, shape)
    if preview_shape:
        _render_shape(img_copy, preview_shape)
    img_resized = cv2.resize(img_copy, None, fx=zoom_level, fy=zoom_level, interpolation=cv2.INTER_LINEAR)
    tk_img = ImageTk.PhotoImage(Image.fromarray(img_resized))
    canvas.create_image(offset_x, offset_y, anchor=tk.NW, image=tk_img)
    canvas.config(scrollregion=canvas.bbox(tk.ALL))

# Function to Handle Mouse Events for Annotations
def start_draw(event):
    global is_drawing, start_x, start_y
    if current_tool in ["circle", "rectangle", "text"]:
        is_drawing = True
        start_x = (event.x - offset_x) / zoom_level
        start_y = (event.y - offset_y) / zoom_level

def draw_preview(event):
    global preview_shape
    if not is_drawing or current_tool not in ["circle", "rectangle"]:
        return
    ex = (event.x - offset_x) / zoom_level
    ey = (event.y - offset_y) / zoom_level
    if current_tool == "circle":
        r = int(((ex - start_x) ** 2 + (ey - start_y) ** 2) ** 0.5 / 2)
        preview_shape = {"type": "circle",
                         "center": (int((start_x + ex) // 2), int((start_y + ey) // 2)),
                         "radius": r, "color": current_color, "thickness": thickness}
    elif current_tool == "rectangle":
        preview_shape = {"type": "rectangle",
                         "start": (int(start_x), int(start_y)),
                         "end": (int(ex), int(ey)),
                         "color": current_color, "thickness": thickness}
    update_canvas()

def stop_draw(event):
    global is_drawing, preview_shape
    preview_shape = None
    if current_tool in ["circle", "rectangle", "text"] and is_drawing:
        is_drawing = False
        end_x = (event.x - offset_x) / zoom_level
        end_y = (event.y - offset_y) / zoom_level

        current_image_path = image_paths[current_image_index]
        if current_tool == "circle":
            radius = int(((end_x - start_x) ** 2 + (end_y - start_y) ** 2) ** 0.5 / 2)
            center_x = int((start_x + end_x) // 2)
            center_y = int((start_y + end_y) // 2)
            annotations_dict[current_image_path].append({
                "type": "circle",
                "center": (int(center_x), int(center_y)),
                "radius": int(radius),
                "color": current_color,
                "thickness": thickness
            })
        
        elif current_tool == "rectangle":
            annotations_dict[current_image_path].append({
                "type": "rectangle",
                "start": (int(start_x), int(start_y)),
                "end": (int(end_x), int(end_y)),
                "color": current_color,
                "thickness": thickness
            })
        
        elif current_tool == "text":
            text = simpledialog.askstring("Text Annotation", "Enter text:")
            if text:
                annotations_dict[current_image_path].append({
                    "type": "text",
                    "text": text,
                    "position": (int(start_x), int(start_y)),
                    "color": current_color,
                    "font_scale": font_size / 12,
                    "thickness": thickness
                })

        update_canvas()

# Function to Clear All Markings
def clear_markings():
    global yolo_processed
    current_image_path = image_paths[current_image_index]
    annotations_dict[current_image_path] = []  # Clear all annotations for the current image
    if current_image_path in yolo_processed:
        yolo_processed.remove(current_image_path) # Allow re-processing by YOLO
    update_canvas()

# Function to Undo Last Annotation
def undo_last_annotation():
    current_image_path = image_paths[current_image_index]
    if annotations_dict[current_image_path]:
        annotations_dict[current_image_path].pop()  # Remove the last annotation
        update_canvas()

# Function to Erase Specific Annotation
def erase_annotation(event):
    current_image_path = image_paths[current_image_index]
    x = (event.x - offset_x) / zoom_level
    y = (event.y - offset_y) / zoom_level

    # Check annotations in reverse order to prioritize user-added annotations
    for shape in reversed(annotations_dict[current_image_path]):
        if shape["type"] == "circle":
            center_x, center_y = shape["center"]
            radius = shape["radius"]
            if (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2:
                annotations_dict[current_image_path].remove(shape)
                break
        elif shape["type"] == "rectangle":
            start_x, start_y = shape["start"]
            end_x, end_y = shape["end"]
            # Ensure correct order for rectangle coordinates
            x_min, x_max = min(start_x, end_x), max(start_x, end_x)
            y_min, y_max = min(start_y, end_y), max(start_y, end_y)
            if x_min <= x <= x_max and y_min <= y <= y_max:
                annotations_dict[current_image_path].remove(shape)
                break
        elif shape["type"] == "text":
            text_x, text_y = shape["position"]
            # A simple heuristic for text-click detection, might need refinement
            if abs(x - text_x) < 50 and abs(y - text_y) < 20: 
                annotations_dict[current_image_path].remove(shape)
                break

    update_canvas()


# Zoom & Pan Functions
def zoom(event):
    global zoom_level
    if event.delta > 0:  # Zoom in
        zoom_level *= 1.1
    else:  # Zoom out
        zoom_level /= 1.1
    update_canvas()

def start_pan(event):
    global last_orig_x, last_orig_y
    last_orig_x = (event.x - offset_x) / zoom_level
    last_orig_y = (event.y - offset_y) / zoom_level

def pan(event):
    global offset_x, offset_y, last_orig_x, last_orig_y
    curr_orig_x = (event.x - offset_x) / zoom_level
    curr_orig_y = (event.y - offset_y) / zoom_level
    offset_x += (curr_orig_x - last_orig_x) * zoom_level
    offset_y += (curr_orig_y - last_orig_y) * zoom_level
    last_orig_x = curr_orig_x
    last_orig_y = curr_orig_y
    update_canvas()

# Function to Choose Annotation Color
def choose_color():
    global current_color
    color = colorchooser.askcolor(title="Choose Annotation Color")[0]
    if color:
        current_color = tuple(int(c) for c in color)
        color_label.config(bg="#%02x%02x%02x" % current_color)  # Update color label

# Function to Set Thickness
def set_thickness(value):
    global thickness
    thickness = int(value)

# Function to Set Font Size
def set_font_size(value):
    global font_size
    font_size = int(value)

# Function to Rotate Image
def rotate_image(degrees):
    global original_img
    if original_img is not None:
        (h, w) = original_img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, degrees, 1.0)
        original_img = cv2.warpAffine(original_img, M, (w, h))
        update_canvas()

# Function to Flip Image
def flip_image(axis):
    global original_img
    if original_img is not None:
        original_img = cv2.flip(original_img, axis)
        update_canvas()

# Function to Start Cropping
def start_crop(event):
    global is_cropping, crop_start_x, crop_start_y
    if current_tool == "crop":
        is_cropping = True
        crop_start_x, crop_start_y = int((event.x - offset_x) / zoom_level), int((event.y - offset_y) / zoom_level)

def stop_crop(event):
    global is_cropping, crop_end_x, crop_end_y
    if current_tool == "crop" and is_cropping:
        is_cropping = False
        crop_end_x, crop_end_y = int((event.x - offset_x) / zoom_level), int((event.y - offset_y) / zoom_level)
        crop_image()

def crop_image():
    global original_img, crop_start_x, crop_start_y, crop_end_x, crop_end_y
    if crop_start_x is not None and crop_start_y is not None and crop_end_x is not None and crop_end_y is not None:
        x1, y1 = min(crop_start_x, crop_end_x), min(crop_start_y, crop_end_y)
        x2, y2 = max(crop_start_x, crop_end_x), max(crop_start_y, crop_end_y)
        original_img = original_img[y1:y2, x1:x2]
        update_canvas()

# Function to Apply Filters
def apply_filter(filter_type):
    global original_img
    if original_img is not None:
        if filter_type == "grayscale":
            original_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)
            original_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2RGB)
        elif filter_type == "blur":
            original_img = cv2.GaussianBlur(original_img, (15, 15), 0)
        elif filter_type == "sharpen":
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            original_img = cv2.filter2D(original_img, -1, kernel)
        elif filter_type == "edge_detection":
            gray = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            original_img = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        elif filter_type == "contrast":
            lab = cv2.cvtColor(original_img, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            lab = cv2.merge((l, a, b))
            original_img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        elif filter_type == "color_thresholding":
            hsv = cv2.cvtColor(original_img, cv2.COLOR_RGB2HSV)
            lower_bound = np.array([0, 50, 50])  # Adjust these values for your needs
            upper_bound = np.array([10, 255, 255])
            mask = cv2.inRange(hsv, lower_bound, upper_bound)
            original_img = cv2.bitwise_and(original_img, original_img, mask=mask)
        elif filter_type == "laplacian":
            gray = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            original_img = cv2.cvtColor(np.uint8(np.absolute(laplacian)), cv2.COLOR_GRAY2RGB)
        elif filter_type == "thermal":
            gray = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)
            thermal = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
            original_img = cv2.cvtColor(thermal, cv2.COLOR_BGR2RGB)
        elif filter_type == "high_pass":
            gray = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)
            kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
            high_pass = cv2.filter2D(gray, -1, kernel)
            original_img = cv2.cvtColor(high_pass, cv2.COLOR_GRAY2RGB)
        update_canvas()

# Function to Reset View
def reset_view():
    global original_img, original_img_state
    if original_img_state is not None:
        original_img = original_img_state.copy()  # Restore the original image
        update_canvas()

# Function to Save All Annotated Images to a ZIP Folder
def save_all_to_zip():
    if not image_paths:
        messagebox.showwarning("No Images", "Please upload images first!")
        return

    try:
        # Ask for desired ZIP size
        desired_size_mb = simpledialog.askinteger("Compression", "Enter desired ZIP size (MB):", minvalue=1, maxvalue=100)
        if not desired_size_mb:
            return

        # Ask for ZIP file name
        zip_name = simpledialog.askstring("ZIP Name", "Enter a name for the ZIP file:")
        if not zip_name:
            return

        desired_size_bytes = desired_size_mb * 1024 * 1024

        # Create a ZIP file
        zip_path = os.path.join(output_dir, f"{zip_name}.zip")
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as zipf:
            current_zip_size = 0
            for i, image_path in enumerate(image_paths):
                # Load the original image
                img = cv2.imread(image_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Draw annotations
                for shape in annotations_dict[image_path]:
                    if shape["type"] == "circle":
                        cv2.circle(img, shape["center"], shape["radius"], shape["color"], shape["thickness"])
                    elif shape["type"] == "rectangle":
                        cv2.rectangle(img, shape["start"], shape["end"], shape["color"], shape["thickness"])
                        if shape.get("source") == "yolo" and shape.get("class_name"):
                            text = shape["class_name"]
                            text_x = shape["start"][0]
                            text_y = shape["start"][1] - 10
                            if text_y < 0: text_y = 10
                            if text_x < 0: text_x = 0
                            cv2.putText(img, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, shape["color"], 2)
                    elif shape["type"] == "text":
                        cv2.putText(img, shape["text"], shape["position"], cv2.FONT_HERSHEY_SIMPLEX, shape["font_scale"], shape["color"], shape["thickness"])

                # Resize the image to reduce file size
                height, width = img.shape[:2]
                max_dimension = 1024  # Set maximum dimension (width or height) to 1024 pixels
                if height > width:
                    new_height = max_dimension
                    new_width = int(width * (max_dimension / height))
                else:
                    new_width = max_dimension
                    new_height = int(height * (max_dimension / width))

                img_resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

                # Save the annotated image with compression
                annotated_path = os.path.join(output_dir, f"annotated_{i}.jpg")
                cv2.imwrite(annotated_path, cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])  # Adjust quality here

                # Check the size of the annotated image
                annotated_size = os.path.getsize(annotated_path)

                # If adding this image exceeds the desired size, stop
                if current_zip_size + annotated_size > desired_size_bytes:
                    messagebox.showwarning("Size Exceeded", "ZIP size exceeds the desired size. Some images may not be included.")
                    break

                # Add the annotated image to the ZIP file
                zipf.write(annotated_path, os.path.basename(annotated_path))
                current_zip_size += annotated_size

        messagebox.showinfo("Success", f"All annotated images saved to:\n{zip_path}")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")

# Function to Set Drawing Tool
def set_tool(tool):
    global current_tool
    current_tool = tool
    # Unbind all mouse events
    canvas.unbind("<ButtonPress-1>")
    canvas.unbind("<ButtonRelease-1>")
    canvas.unbind("<B1-Motion>")
    canvas.unbind("<Button-3>")

    # Bind events based on the selected tool
    if tool in ["circle", "rectangle", "text"]:
        canvas.bind("<ButtonPress-1>", start_draw)
        canvas.bind("<ButtonRelease-1>", stop_draw)
        if tool in ["circle", "rectangle"]:
            canvas.bind("<B1-Motion>", draw_preview)
    elif tool == "crop":
        canvas.bind("<ButtonPress-1>", start_crop)
        canvas.bind("<ButtonRelease-1>", stop_crop)
    canvas.bind("<Button-3>", erase_annotation)  # Right-click to erase

# Function to Navigate Between Images
def next_image():
    global current_image_index
    if current_image_index < len(image_paths) - 1:
        current_image_index += 1
        load_current_image()
        update_image_counter()

def prev_image():
    global current_image_index
    if current_image_index > 0:
        current_image_index -= 1
        load_current_image()
        update_image_counter()

# Function to Update Image Counter
def update_image_counter():
    counter_label.config(text=f"Image {current_image_index + 1} of {len(image_paths)}")

# Function to Show How to Use Guide
def show_how_to_use():
    guide = """
    **How to Use the Bulk Image Annotation & Compression Tool**

    1. **Upload Images**:
       - Click the "Upload Images" button.
       - Select one or more images from your computer.

    2. **Navigate Between Images**:
       - Use the "Previous" and "Next" buttons to switch between images.

    3. **Annotate Images**:
       - Use the "Circle", "Rectangle", or "Text" tools to draw annotations.
       - Click the "Choose Color" button to pick a color.
       - Adjust the thickness of shapes and text using the sliders.

    4. **Detect Damage (YOLO)**:
       - Click "Detect Damage" to run the YOLO model on the current image.
       - Adjust "Confidence Threshold" and "IOU Threshold" to fine-tune detection.

    5. **Edit Annotations**:
       - Click "Clear All Markings" to remove all annotations (including YOLO detections).
       - Click "Undo Last Annotation" to remove the last annotation.
       - Right-click on an annotation to erase it.

    6. **Zoom and Pan**:
       - Use the mouse wheel to zoom in or out.
       - Press and hold the middle mouse button to pan.

    7. **Save Annotated Images**:
       - Click "Save All to ZIP".
       - Enter the desired ZIP size and name.
       - The annotated images will be saved in a ZIP file.

    **Tips**:
    - Use zoom and pan to work on detailed areas.
    - Right-click to erase specific annotations.
    """
    messagebox.showinfo("How to Use", guide)

# Function to update confidence threshold
def set_confidence_threshold(value):
    global confidence_threshold
    confidence_threshold = float(value)

# Function to update IOU threshold
def set_iou_threshold(value):
    global iou_threshold
    iou_threshold = float(value)

# Function to Open Compare View
def open_compare_view():
    global compare_mode, compare_index_1, compare_index_2

    # Ask the user to select two images for comparison
    compare_index_1 = simpledialog.askinteger("Compare View", "Enter the index of the first image (1-based):")
    compare_index_2 = simpledialog.askinteger("Compare View", "Enter the index of the second image (1-based):")

    if compare_index_1 is None or compare_index_2 is None:
        return

    # Validate indices
    if compare_index_1 < 1 or compare_index_1 > len(image_paths) or compare_index_2 < 1 or compare_index_2 > len(image_paths):
        messagebox.showerror("Error", "Invalid image indices!")
        return

    # Convert to 0-based indices
    compare_index_1 -= 1
    compare_index_2 -= 1

    # Open a new window for comparison
    compare_window = tk.Toplevel(root)
    compare_window.title("Compare View")
    compare_window.geometry("1200x600")

    # Load the two images
    img1 = cv2.imread(image_paths[compare_index_1])
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.imread(image_paths[compare_index_2])
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    # Resize images to fit the window
    img1_resized = cv2.resize(img1, (500, 500))
    img2_resized = cv2.resize(img2, (500, 500))

    # Convert to PhotoImage
    tk_img1 = ImageTk.PhotoImage(Image.fromarray(img1_resized))
    tk_img2 = ImageTk.PhotoImage(Image.fromarray(img2_resized))

    # Create canvases for the two images
    canvas1 = tk.Canvas(compare_window, width=500, height=500, bg="white")
    canvas1.pack(side=tk.LEFT, padx=10, pady=10)
    canvas1.create_image(0, 0, anchor=tk.NW, image=tk_img1)

    canvas2 = tk.Canvas(compare_window, width=500, height=500, bg="white")
    canvas2.pack(side=tk.RIGHT, padx=10, pady=10)
    canvas2.create_image(0, 0, anchor=tk.NW, image=tk_img2)

    # Keep references to the images to prevent garbage collection
    canvas1.image = tk_img1
    canvas2.image = tk_img2


# ── Color scheme ──────────────────────────────────────────────────────────────
C_BG      = "#1e1e1e"
C_SIDEBAR = "#252526"
C_SECTION = "#2d2d2d"
C_ACCENT  = "#0078d4"
C_ACCENT2 = "#005a9e"
C_TEXT    = "#cccccc"
C_TEXT_HI = "#ffffff"
C_BTN     = "#3c3c3c"
C_BTN_ACT = "#094771"
C_SEP     = "#3e3e42"

# ── Root window ───────────────────────────────────────────────────────────────
root = tk.Tk()
root.title("Car Damage Annotation Tool")
root.geometry("1280x760")
root.configure(bg=C_BG)
root.minsize(900, 600)

# ── Top toolbar ───────────────────────────────────────────────────────────────
toolbar = tk.Frame(root, bg=C_SIDEBAR, height=48)
toolbar.pack(side=tk.TOP, fill=tk.X)
toolbar.pack_propagate(False)

tk.Label(toolbar, text="  Car Damage Annotation Tool",
         bg=C_SIDEBAR, fg=C_TEXT_HI,
         font=("Segoe UI", 11, "bold")).pack(side=tk.LEFT, padx=8)

def _tb_btn(text, cmd):
    b = tk.Button(toolbar, text=text, command=cmd, bg=C_BTN, fg=C_TEXT_HI,
                  relief="flat", padx=14, pady=10, cursor="hand2", bd=0,
                  activebackground=C_ACCENT, activeforeground=C_TEXT_HI,
                  font=("Segoe UI", 9))
    b.pack(side=tk.RIGHT, padx=2, pady=4)
    return b

_tb_btn("?  Help",          show_how_to_use)
_tb_btn("⊞  Compare",       open_compare_view)
_tb_btn("💾  Save to ZIP",  save_all_to_zip)
_tb_btn("📂  Upload Images", load_images)

# ── Main layout ───────────────────────────────────────────────────────────────
main_frame = tk.Frame(root, bg=C_BG)
main_frame.pack(fill=tk.BOTH, expand=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
sidebar = tk.Frame(main_frame, bg=C_SIDEBAR, width=220)
sidebar.pack(side=tk.LEFT, fill=tk.Y)
sidebar.pack_propagate(False)

sb_canvas = tk.Canvas(sidebar, bg=C_SIDEBAR, highlightthickness=0, width=220, bd=0)
sb_scroll  = tk.Scrollbar(sidebar, orient="vertical", command=sb_canvas.yview)
sb_frame   = tk.Frame(sb_canvas, bg=C_SIDEBAR)

sb_frame.bind("<Configure>",
    lambda e: sb_canvas.configure(scrollregion=sb_canvas.bbox("all")))
sb_canvas.create_window((0, 0), window=sb_frame, anchor="nw")
sb_canvas.configure(yscrollcommand=sb_scroll.set)
sb_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
sb_scroll.pack(side=tk.RIGHT, fill=tk.Y)

def _sb_scroll_handler(e):
    sb_canvas.yview_scroll(-1 * (e.delta // 120), "units")

# ── Sidebar helpers ───────────────────────────────────────────────────────────
def _section(title):
    tk.Frame(sb_frame, bg=C_SEP, height=1).pack(fill=tk.X, padx=8, pady=(14, 0))
    tk.Label(sb_frame, text=title.upper(), bg=C_SIDEBAR, fg="#6e6e6e",
             font=("Segoe UI", 7, "bold")).pack(anchor="w", padx=12, pady=(4, 2))

def _sb_btn(text, cmd, bg=None):
    b = tk.Button(sb_frame, text=text, command=cmd,
                  bg=bg or C_BTN, fg=C_TEXT_HI, relief="flat",
                  anchor="w", padx=12, pady=6, cursor="hand2", bd=0,
                  activebackground=C_ACCENT, activeforeground=C_TEXT_HI,
                  font=("Segoe UI", 9), width=24)
    b.pack(fill=tk.X, padx=8, pady=1)
    b.bind("<MouseWheel>", _sb_scroll_handler)
    return b

def _sb_row(pairs):
    row = tk.Frame(sb_frame, bg=C_SIDEBAR)
    row.pack(fill=tk.X, padx=8, pady=1)
    row.bind("<MouseWheel>", _sb_scroll_handler)
    for text, cmd in pairs:
        b = tk.Button(row, text=text, command=cmd, bg=C_BTN, fg=C_TEXT_HI,
                      relief="flat", padx=4, pady=5, cursor="hand2", bd=0,
                      activebackground=C_ACCENT, activeforeground=C_TEXT_HI,
                      font=("Segoe UI", 8))
        b.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)
        b.bind("<MouseWheel>", _sb_scroll_handler)

def _slider(label, from_, to_, res, init, cmd):
    tk.Label(sb_frame, text=label, bg=C_SIDEBAR, fg=C_TEXT,
             font=("Segoe UI", 8)).pack(anchor="w", padx=12, pady=(6, 0))
    s = tk.Scale(sb_frame, from_=from_, to=to_, resolution=res,
                 orient=tk.HORIZONTAL, command=cmd,
                 bg=C_SIDEBAR, fg=C_TEXT, troughcolor=C_SECTION,
                 activebackground=C_ACCENT, highlightthickness=0,
                 sliderrelief="flat", bd=0, font=("Segoe UI", 8))
    s.set(init)
    s.pack(fill=tk.X, padx=8)
    s.bind("<MouseWheel>", _sb_scroll_handler)
    return s

# ── Drawing tools ─────────────────────────────────────────────────────────────
_section("Drawing Tools")

tool_btns = {}

def _tool_cmd(tool):
    def cmd():
        set_tool(tool)
        for t, b in tool_btns.items():
            b.config(bg=C_BTN_ACT if t == tool else C_BTN)
    return cmd

tool_btns["circle"]    = _sb_btn("⭕  Circle",    _tool_cmd("circle"))
tool_btns["rectangle"] = _sb_btn("⬛  Rectangle", _tool_cmd("rectangle"))
tool_btns["text"]      = _sb_btn("T    Text",     _tool_cmd("text"))
tool_btns["crop"]      = _sb_btn("✂   Crop",      _tool_cmd("crop"))

# ── Annotations ───────────────────────────────────────────────────────────────
_section("Annotations")
_sb_row([("↩ Undo", undo_last_annotation), ("✕ Clear All", clear_markings)])

# ── Style ─────────────────────────────────────────────────────────────────────
_section("Style")

color_row = tk.Frame(sb_frame, bg=C_SIDEBAR)
color_row.pack(fill=tk.X, padx=8, pady=(4, 2))
color_row.bind("<MouseWheel>", _sb_scroll_handler)
tk.Label(color_row, text="Color", bg=C_SIDEBAR, fg=C_TEXT,
         font=("Segoe UI", 9)).pack(side=tk.LEFT, padx=4)
color_label = tk.Label(color_row, bg="#%02x%02x%02x" % current_color,
                        width=4, cursor="hand2", relief="flat")
color_label.pack(side=tk.RIGHT, padx=4)
tk.Button(color_row, text="Pick", command=choose_color,
          bg=C_BTN, fg=C_TEXT_HI, relief="flat", padx=8, pady=2,
          cursor="hand2", bd=0, font=("Segoe UI", 8),
          activebackground=C_ACCENT).pack(side=tk.RIGHT, padx=2)

thickness_slider = _slider("Thickness", 1,  10, 1,    thickness,            set_thickness)
font_size_slider  = _slider("Font Size", 10, 50, 1,    font_size,            set_font_size)

# ── Image operations ──────────────────────────────────────────────────────────
_section("Image")
_sb_row([("↻ 90°",  lambda: rotate_image(90)),  ("↻ 180°", lambda: rotate_image(180))])
_sb_row([("↻ 270°", lambda: rotate_image(270)), ("↔ Flip H", lambda: flip_image(1))])
_sb_btn("↕  Flip Vertical", lambda: flip_image(0))
_sb_btn("⟲  Reset View",    reset_view)

# ── Filters ───────────────────────────────────────────────────────────────────
_section("Filters")
_sb_row([("Grayscale",  lambda: apply_filter("grayscale")),
         ("Blur",       lambda: apply_filter("blur"))])
_sb_row([("Sharpen",    lambda: apply_filter("sharpen")),
         ("Edges",      lambda: apply_filter("edge_detection"))])
_sb_row([("Contrast",   lambda: apply_filter("contrast")),
         ("Threshold",  lambda: apply_filter("color_thresholding"))])
_sb_row([("Laplacian",  lambda: apply_filter("laplacian")),
         ("Thermal",    lambda: apply_filter("thermal"))])
_sb_btn("High-Pass", lambda: apply_filter("high_pass"))

# ── AI detection ──────────────────────────────────────────────────────────────
_section("AI Detection")
detect_btn = tk.Button(sb_frame, text="⚡  Detect Damage", command=run_yolo_detection,
                        bg=C_ACCENT, fg=C_TEXT_HI, relief="flat", padx=12, pady=8, bd=0,
                        cursor="hand2", activebackground=C_ACCENT2, activeforeground=C_TEXT_HI,
                        font=("Segoe UI", 9, "bold"))
detect_btn.pack(fill=tk.X, padx=8, pady=(4, 2))

confidence_slider = _slider("Confidence",    0.0, 1.0, 0.05, confidence_threshold, set_confidence_threshold)
iou_slider        = _slider("IOU Threshold", 0.0, 1.0, 0.05, iou_threshold,        set_iou_threshold)

tk.Frame(sb_frame, bg=C_SIDEBAR, height=16).pack()

sb_canvas.bind("<MouseWheel>", _sb_scroll_handler)
sb_frame.bind("<MouseWheel>",  _sb_scroll_handler)

# ── Canvas area ───────────────────────────────────────────────────────────────
canvas_area = tk.Frame(main_frame, bg=C_BG)
canvas_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

canvas = tk.Canvas(canvas_area, bg="#141414", highlightthickness=0, cursor="crosshair")
canvas.pack(fill=tk.BOTH, expand=True)

canvas.bind("<MouseWheel>",    zoom)
canvas.bind("<ButtonPress-2>", start_pan)
canvas.bind("<B2-Motion>",     pan)
canvas.bind("<Button-3>",      erase_annotation)

# ── Status bar ────────────────────────────────────────────────────────────────
status_bar = tk.Frame(root, bg=C_SIDEBAR, height=34)
status_bar.pack(side=tk.BOTTOM, fill=tk.X)
status_bar.pack_propagate(False)

counter_label = tk.Label(status_bar, text="No images loaded",
                          bg=C_SIDEBAR, fg=C_TEXT, font=("Segoe UI", 9))
counter_label.pack(side=tk.LEFT, padx=16, pady=6)

nav_frame = tk.Frame(status_bar, bg=C_SIDEBAR)
nav_frame.pack(side=tk.RIGHT, padx=8, pady=5)
tk.Button(nav_frame, text="◀  Prev", command=prev_image,
          bg=C_BTN, fg=C_TEXT_HI, relief="flat", padx=10, pady=3,
          cursor="hand2", activebackground=C_ACCENT, bd=0,
          font=("Segoe UI", 9)).pack(side=tk.LEFT, padx=2)
tk.Button(nav_frame, text="Next  ▶", command=next_image,
          bg=C_BTN, fg=C_TEXT_HI, relief="flat", padx=10, pady=3,
          cursor="hand2", activebackground=C_ACCENT, bd=0,
          font=("Segoe UI", 9)).pack(side=tk.LEFT, padx=2)

hints = ("Scroll = zoom  |  Middle-drag = pan  |  Right-click = erase  |  "
         "1-4 = tools  |  Ctrl+Z = undo  |  Del = clear  |  ← → = navigate")
tk.Label(status_bar, text=hints, bg=C_SIDEBAR, fg="#555565",
         font=("Segoe UI", 8)).pack(side=tk.LEFT, padx=24)

# ── Keyboard shortcuts ────────────────────────────────────────────────────────
root.bind("<Control-z>", lambda e: undo_last_annotation())
root.bind("<Control-o>", lambda e: load_images())
root.bind("<Control-s>", lambda e: save_all_to_zip())
root.bind("<Delete>",    lambda e: clear_markings())
root.bind("<Left>",      lambda e: prev_image())
root.bind("<Right>",     lambda e: next_image())
root.bind("1", lambda e: _tool_cmd("circle")())
root.bind("2", lambda e: _tool_cmd("rectangle")())
root.bind("3", lambda e: _tool_cmd("text")())
root.bind("4", lambda e: _tool_cmd("crop")())

# ── Initialise ────────────────────────────────────────────────────────────────
set_tool("circle")
tool_btns["circle"].config(bg=C_BTN_ACT)

root.mainloop()
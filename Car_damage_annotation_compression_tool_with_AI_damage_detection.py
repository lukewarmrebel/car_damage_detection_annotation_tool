import tkinter as tk
from tkinter import filedialog, messagebox, colorchooser, simpledialog
import cv2
from PIL import Image, ImageTk
import numpy as np
import os
import zipfile
from ultralytics import YOLO

# Global Variables
image_paths = []  # List to store paths of all uploaded images
current_image_index = 0  # Index of the currently displayed image
output_dir = "processed_images"
os.makedirs(output_dir, exist_ok=True)

# YOLOv8 Model
# Ensure the path to your best.pt model is correct.
model = YOLO(r'C:\Users\ADMIN\Desktop\Office stuff\FT BI tool\Car_damage_annotation_compression_tool\best.pt')  # Load YOLOv8 model
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
confidence_threshold = 0.5  # Default confidence threshold
iou_threshold = 0.5       # Default IOU threshold

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
    
    # Check if the image has already been processed by YOLO in this session
    if image_path in yolo_processed:
        messagebox.showinfo("Already Processed", "This image has already been processed by YOLO in this session. Clear markings to re-run.")
        return

    # To run YOLO, we need the original, unprocessed image.
    yolo_img = cv2.imread(image_path)
    yolo_img = cv2.cvtColor(yolo_img, cv2.COLOR_BGR2RGB)

    results = model.predict(yolo_img, conf=confidence_threshold, iou=iou_threshold)
    
    # Get class names from the model
    class_names = model.names

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            class_id = int(box.cls[0])
            class_name = class_names[class_id]

            annotations_dict[image_path].append({
                "type": "rectangle",
                "start": (x1, y1),
                "end": (x2, y2),
                "color": (0, 255, 0),  # Green for YOLO detections
                "thickness": thickness,
                "source": "yolo",
                "class_name": class_name  # Store the class name
            })
    yolo_processed.add(image_path) # Mark as processed
    update_canvas()


# Function to Update Canvas
def update_canvas():
    global tk_img, img
    img_copy = original_img.copy()

    # Redraw all saved annotations for the current image
    current_image_path = image_paths[current_image_index]
    for shape in annotations_dict[current_image_path]:
        if shape["type"] == "circle":
            cv2.circle(img_copy, shape["center"], shape["radius"], shape["color"], shape["thickness"])
        elif shape["type"] == "rectangle":
            cv2.rectangle(img_copy, shape["start"], shape["end"], shape["color"], shape["thickness"])
            if shape.get("source") == "yolo" and shape.get("class_name"):
                text = shape["class_name"]
                # Position the text slightly above the top-left corner of the bounding box
                text_x = shape["start"][0]
                text_y = shape["start"][1] - 10  # 10 pixels above the top edge
                # Ensure text doesn't go out of bounds
                if text_y < 0: text_y = 10
                if text_x < 0: text_x = 0

                cv2.putText(img_copy, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, shape["color"], 2)
        elif shape["type"] == "text":
            cv2.putText(img_copy, shape["text"], shape["position"], cv2.FONT_HERSHEY_SIMPLEX, shape["font_scale"], shape["color"], shape["thickness"])

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

def stop_draw(event):
    global is_drawing
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


# GUI Setup
root = tk.Tk()
root.title("Bulk Image Annotation & Compression Tool")
root.geometry("900x700")

# Main Frame for Buttons and Controls
control_frame = tk.Frame(root)
control_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)

# Upload Button
upload_btn = tk.Button(control_frame, text="Upload Images", command=load_images)
upload_btn.grid(row=0, column=0, padx=5, pady=5)

# Tool Selection
circle_btn = tk.Button(control_frame, text="Circle", command=lambda: set_tool("circle"))
circle_btn.grid(row=0, column=1, padx=5, pady=5)

rect_btn = tk.Button(control_frame, text="Rectangle", command=lambda: set_tool("rectangle"))
rect_btn.grid(row=0, column=2, padx=5, pady=5)

text_btn = tk.Button(control_frame, text="Text", command=lambda: set_tool("text"))
text_btn.grid(row=0, column=3, padx=5, pady=5)

color_btn = tk.Button(control_frame, text="Choose Color", command=choose_color)
color_btn.grid(row=0, column=4, padx=5, pady=5)

# Color Label
color_label = tk.Label(control_frame, text="   ", bg="#%02x%02x%02x" % current_color)
color_label.grid(row=0, column=5, padx=5, pady=5)

# Thickness Slider
tk.Label(control_frame, text="Thickness:").grid(row=1, column=0, padx=5, pady=5)
thickness_slider = tk.Scale(control_frame, from_=1, to=10, orient=tk.HORIZONTAL, command=set_thickness)
thickness_slider.set(thickness)
thickness_slider.grid(row=1, column=1, columnspan=2, padx=5, pady=5)

# Font Size Slider
tk.Label(control_frame, text="Font Size:").grid(row=1, column=3, padx=5, pady=5)
font_size_slider = tk.Scale(control_frame, from_=10, to=50, orient=tk.HORIZONTAL, command=set_font_size)
font_size_slider.set(font_size)
font_size_slider.grid(row=1, column=4, columnspan=2, padx=5, pady=5)

# Image Counter
counter_label = tk.Label(control_frame, text="Image 0 of 0")
counter_label.grid(row=2, column=0, columnspan=2, padx=5, pady=5)

# Navigation Buttons
prev_btn = tk.Button(control_frame, text="Previous", command=prev_image)
prev_btn.grid(row=2, column=2, padx=5, pady=5)

next_btn = tk.Button(control_frame, text="Next", command=next_image)
next_btn.grid(row=2, column=3, padx=5, pady=5)

# Clear and Undo Buttons
clear_btn = tk.Button(control_frame, text="Clear All Markings", command=clear_markings)
clear_btn.grid(row=2, column=4, padx=5, pady=5)

undo_btn = tk.Button(control_frame, text="Undo Last Annotation", command=undo_last_annotation)
undo_btn.grid(row=2, column=5, padx=5, pady=5)

# YOLO Detection Button
detect_damage_btn = tk.Button(control_frame, text="Detect Damage (YOLO)", command=run_yolo_detection)
detect_damage_btn.grid(row=3, column=0, columnspan=2, padx=5, pady=5)

# Confidence Threshold Slider
tk.Label(control_frame, text="Conf. Threshold:").grid(row=3, column=2, padx=5, pady=5)
confidence_slider = tk.Scale(control_frame, from_=0.0, to=1.0, resolution=0.05, orient=tk.HORIZONTAL, command=set_confidence_threshold)
confidence_slider.set(confidence_threshold)
confidence_slider.grid(row=3, column=3, padx=5, pady=5)

# IOU Threshold Slider
tk.Label(control_frame, text="IOU Threshold:").grid(row=3, column=4, padx=5, pady=5)
iou_slider = tk.Scale(control_frame, from_=0.0, to=1.0, resolution=0.05, orient=tk.HORIZONTAL, command=set_iou_threshold)
iou_slider.set(iou_threshold)
iou_slider.grid(row=3, column=5, padx=5, pady=5)

# Save All Button
save_all_btn = tk.Button(control_frame, text="Save All to ZIP", command=save_all_to_zip)
save_all_btn.grid(row=4, column=0, columnspan=3, padx=5, pady=5)

# How to Use Button
how_to_use_btn = tk.Button(control_frame, text="How to Use", command=show_how_to_use)
how_to_use_btn.grid(row=4, column=3, columnspan=3, padx=5, pady=5)

# Rotation and Flipping Buttons
rotate_90_btn = tk.Button(control_frame, text="Rotate 90°", command=lambda: rotate_image(90))
rotate_90_btn.grid(row=5, column=0, padx=5, pady=5)

rotate_180_btn = tk.Button(control_frame, text="Rotate 180°", command=lambda: rotate_image(180))
rotate_180_btn.grid(row=5, column=1, padx=5, pady=5)

rotate_270_btn = tk.Button(control_frame, text="Rotate 270°", command=lambda: rotate_image(270))
rotate_270_btn.grid(row=5, column=2, padx=5, pady=5)

flip_horizontal_btn = tk.Button(control_frame, text="Flip Horizontal", command=lambda: flip_image(1))
flip_horizontal_btn.grid(row=5, column=3, padx=5, pady=5)

flip_vertical_btn = tk.Button(control_frame, text="Flip Vertical", command=lambda: flip_image(0))
flip_vertical_btn.grid(row=5, column=4, padx=5, pady=5)

# Cropping Buttons
crop_btn = tk.Button(control_frame, text="Crop", command=lambda: set_tool("crop"))
crop_btn.grid(row=6, column=0, padx=5, pady=5)

# Filter Buttons
grayscale_btn = tk.Button(control_frame, text="Grayscale", command=lambda: apply_filter("grayscale"))
grayscale_btn.grid(row=6, column=1, padx=5, pady=5)

blur_btn = tk.Button(control_frame, text="Blur", command=lambda: apply_filter("blur"))
blur_btn.grid(row=6, column=2, padx=5, pady=5)

sharpen_btn = tk.Button(control_frame, text="Sharpen", command=lambda: apply_filter("sharpen"))
sharpen_btn.grid(row=6, column=3, padx=5, pady=5)

edge_detection_btn = tk.Button(control_frame, text="Edge Detection", command=lambda: apply_filter("edge_detection"))
edge_detection_btn.grid(row=6, column=4, padx=5, pady=5)

contrast_btn = tk.Button(control_frame, text="Contrast", command=lambda: apply_filter("contrast"))
contrast_btn.grid(row=6, column=5, padx=5, pady=5)

color_threshold_btn = tk.Button(control_frame, text="Color Threshold", command=lambda: apply_filter("color_thresholding"))
color_threshold_btn.grid(row=7, column=0, padx=5, pady=5)

laplacian_btn = tk.Button(control_frame, text="Laplacian", command=lambda: apply_filter("laplacian"))
laplacian_btn.grid(row=7, column=1, padx=5, pady=5)

thermal_btn = tk.Button(control_frame, text="Thermal", command=lambda: apply_filter("thermal"))
thermal_btn.grid(row=7, column=2, padx=5, pady=5)

high_pass_btn = tk.Button(control_frame, text="High-Pass", command=lambda: apply_filter("high_pass"))
high_pass_btn.grid(row=7, column=3, padx=5, pady=5)

# Reset View Button
reset_view_btn = tk.Button(control_frame, text="Reset View", command=reset_view)
reset_view_btn.grid(row=8, column=0, columnspan=6, padx=5, pady=5)

# Compare View Button
compare_view_btn = tk.Button(control_frame, text="Compare View", command=open_compare_view)
compare_view_btn.grid(row=9, column=0, columnspan=6, padx=5, pady=5)

# Canvas for Image Display
canvas_frame = tk.Frame(root)
canvas_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

canvas = tk.Canvas(canvas_frame, width=800, height=500, bg="white")
canvas.pack(fill=tk.BOTH, expand=True)

# Bind Mouse Events
canvas.bind("<MouseWheel>", zoom)
canvas.bind("<ButtonPress-2>", start_pan)  # Middle mouse button press
canvas.bind("<B2-Motion>", pan)  # Middle mouse button drag
canvas.bind("<Button-3>", erase_annotation)  # Right-click to erase

# Set default tool to circle
set_tool("circle")

# Run GUI
root.mainloop()
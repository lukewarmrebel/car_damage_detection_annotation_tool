import tkinter as tk
from tkinter import filedialog, messagebox, colorchooser, simpledialog
import cv2
from PIL import Image, ImageTk
import numpy as np
import os
import zipfile
from ultralytics import YOLO # Import YOLO from ultralytics

# Global Variables
image_paths = []  # List to store paths of all uploaded images
current_image_index = 0  # Index of the currently displayed image
output_dir = "processed_images"
os.makedirs(output_dir, exist_ok=True)

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

# NEW: Global variable for the YOLO detection model
detection_model = None

# NEW: Function to load the local detection model
def load_detection_model():
    global detection_model
    try:
        model_path = r'C:\Users\ADMIN\Desktop\Office stuff\FT BI tool\Car_damage_annotation_compression_tool\best.pt' # Path to your downloaded safetensors file
        detection_model = YOLO(model_path)
        print(f"Damage detection model loaded from {model_path}")
        # messagebox.showinfo("Model Loaded", f"Damage detection model loaded successfully from {model_path}")
    except Exception as e:
        messagebox.showerror("Model Load Error", f"Failed to load detection model: {e}\n"\
                                                  "Please ensure 'ultralytics' and 'safetensors' are installed and the model file exists at the correct path.")

# Function to Load Bulk Images
def load_images():
    global image_paths, current_image_index, annotations_dict, original_img_state
    files = filedialog.askopenfilenames(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    if not files:
        return

    image_paths = list(files)
    current_image_index = 0
    annotations_dict = {path: [] for path in image_paths}  # Initialize annotations for each image
    load_current_image()
    update_image_counter()

# Function to Load the Current Image
def load_current_image():
    global image_paths, current_image_index, img, original_img, tk_img, zoom_level, offset_x, offset_y, original_img_state
    if not image_paths:
        canvas.delete("all") # Clear canvas if no images
        img = None # Clear image variable
        original_img = None
        original_img_state = None
        update_image_counter()
        return

    image_path = image_paths[current_image_index]
    img = cv2.imread(image_path)
    if img is None:
        messagebox.showerror("Error", f"Failed to load image: {image_path}")
        # Attempt to remove invalid path and load next
        del image_paths[current_image_index]
        if current_image_index >= len(image_paths) and len(image_paths) > 0:
            current_image_index = len(image_paths) - 1
        elif len(image_paths) == 0:
            current_image_index = 0 # No images left
        load_current_image() # Try loading again
        return

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    original_img = img.copy()
    original_img_state = original_img.copy()  # Save the original state of the image
    zoom_level = 1.0
    offset_x, offset_y = 0, 0
    update_canvas()

# Function to Update Canvas
def update_canvas():
    global tk_img, img
    if original_img is None: # Handle case where no image is loaded
        canvas.delete("all")
        return

    img_copy = original_img.copy()

    # Redraw all saved annotations for the current image
    if image_paths:
        current_image_path = image_paths[current_image_index]
        for shape in annotations_dict[current_image_path]:
            if shape["type"] == "circle":
                cv2.circle(img_copy, shape["center"], shape["radius"], shape["color"], shape["thickness"])
            elif shape["type"] == "rectangle":
                cv2.rectangle(img_copy, shape["start"], shape["end"], shape["color"], shape["thickness"])
            elif shape["type"] == "text":
                # Ensure color is BGR for cv2.putText if it expects it, though RGB is generally fine with PIL conversion.
                # Using current_color directly from annotation_dict which is already RGB
                cv2.putText(img_copy, shape["text"], shape["position"], cv2.FONT_HERSHEY_SIMPLEX, shape["font_scale"], shape["color"], shape["thickness"])

    img_resized = cv2.resize(img_copy, None, fx=zoom_level, fy=zoom_level, interpolation=cv2.INTER_LINEAR)
    tk_img = ImageTk.PhotoImage(Image.fromarray(img_resized))
    
    canvas.delete("all") # Clear previous image before drawing new one
    canvas.create_image(offset_x, offset_y, anchor=tk.NW, image=tk_img)
    canvas.config(scrollregion=canvas.bbox(tk.ALL))

# Function to Handle Mouse Events for Annotations
def start_draw(event):
    global is_drawing, start_x, start_y
    if current_tool in ["circle", "rectangle", "text"]:
        is_drawing = True
        start_x, start_y = int((event.x - offset_x) / zoom_level), int((event.y - offset_y) / zoom_level)

def stop_draw(event):
    global is_drawing
    if current_tool in ["circle", "rectangle", "text"] and is_drawing:
        is_drawing = False
        end_x, end_y = int((event.x - offset_x) / zoom_level), int((event.y - offset_y) / zoom_level)

        current_image_path = image_paths[current_image_index]
        if current_tool == "circle":
            radius = int(((end_x - start_x) ** 2 + (end_y - start_y) ** 2) ** 0.5 / 2)
            center_x = (start_x + end_x) // 2
            center_y = (start_y + end_y) // 2
            annotations_dict[current_image_path].append({
                "type": "circle",
                "center": (center_x, center_y),
                "radius": radius,
                "color": current_color,
                "thickness": thickness
            })
            
        elif current_tool == "rectangle":
            annotations_dict[current_image_path].append({
                "type": "rectangle",
                "start": (start_x, start_y),
                "end": (end_x, end_y),
                "color": current_color,
                "thickness": thickness
            })
            
        elif current_tool == "text":
            text = simpledialog.askstring("Text Annotation", "Enter text:")
            if text:
                annotations_dict[current_image_path].append({
                    "type": "text",
                    "text": text,
                    "position": (start_x, start_y),
                    "color": current_color,
                    "font_scale": font_size / 12,  # Scale based on default font size
                    "thickness": thickness
                })

        update_canvas()

# NEW: Function to perform damage detection and annotate
def detect_damage_and_annotate():
    global detection_model, original_img, annotations_dict
    if not image_paths:
        messagebox.showwarning("No Image", "Please upload an image first!")
        return
    
    if detection_model is None:
        messagebox.showerror("Model Not Loaded", "Damage detection model is not loaded. Check console for errors.")
        return

    current_image_path = image_paths[current_image_index]

    # Clear existing annotations before adding new detections
    annotations_dict[current_image_path] = []

    try:
        # Convert original_img (RGB numpy array) to PIL Image for prediction if needed,
        # or directly pass the numpy array if YOLO.predict supports it.
        # YOLO.predict can directly take numpy arrays or file paths.
        results = detection_model.predict(source=original_img, save=False, verbose=False) # Predict on the current image

        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()  # Bounding box coordinates
            names = r.names # Class names map
            classes = r.boxes.cls.cpu().numpy() # Class IDs

            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box)
                class_id = int(classes[i])
                class_name = names[class_id]

                # Add rectangle annotation
                annotations_dict[current_image_path].append({
                    "type": "rectangle",
                    "start": (x1, y1),
                    "end": (x2, y2),
                    "color": (0, 255, 0), # Green color for detected damages
                    "thickness": 2
                })

                # Add text annotation (class name)
                # Position text above the box or inside if space is limited
                text_pos = (x1, y1 - 10) if y1 - 10 > 0 else (x1, y1 + 20)
                annotations_dict[current_image_path].append({
                    "type": "text",
                    "text": class_name,
                    "position": text_pos,
                    "color": (0, 255, 0), # Green color for text
                    "font_scale": 0.6, # Adjusted font scale for better visibility
                    "thickness": 1
                })
        
        messagebox.showinfo("Detection Complete", f"Detected damages on Image {current_image_index + 1}.")

    except Exception as e:
        messagebox.showerror("Detection Error", f"An error occurred during damage detection: {e}")

    update_canvas() # Update canvas to show new annotations


# Function to Clear All Markings
def clear_markings():
    if not image_paths:
        return
    current_image_path = image_paths[current_image_index]
    annotations_dict[current_image_path] = []  # Clear all annotations for the current image
    update_canvas()

# Function to Undo Last Annotation
def undo_last_annotation():
    if not image_paths:
        return
    current_image_path = image_paths[current_image_index]
    if annotations_dict[current_image_path]:
        annotations_dict[current_image_path].pop()  # Remove the last annotation
    update_canvas()

# Function to Erase Specific Annotation
def erase_annotation(event):
    if not image_paths:
        return
    current_image_path = image_paths[current_image_index]
    x, y = int((event.x - offset_x) / zoom_level), int((event.y - offset_y) / zoom_level)

    # Check if the click is inside any annotation
    for shape in annotations_dict[current_image_path]:
        if shape["type"] == "circle":
            center_x, center_y = shape["center"]
            radius = shape["radius"]
            if (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2:
                annotations_dict[current_image_path].remove(shape)
                break
        elif shape["type"] == "rectangle":
            start_x, start_y = shape["start"]
            end_x, end_y = shape["end"]
            # Ensure start < end for proper comparison
            x_min, x_max = min(start_x, end_x), max(start_x, end_x)
            y_min, y_max = min(start_y, end_y), max(start_y, end_y)
            if x_min <= x <= x_max and y_min <= y <= y_max:
                annotations_dict[current_image_path].remove(shape)
                break
        elif shape["type"] == "text":
            text_x, text_y = shape["position"]
            # Approximate text bounding box (you can improve this logic based on cv2.getTextSize)
            # For simplicity, using a fixed-size approximation
            if abs(x - text_x) < 50 and abs(y - text_y) < 20: # This is a rough approximation
                annotations_dict[current_image_path].remove(shape)
                break

    update_canvas()

# Zoom & Pan Functions
def zoom(event):
    global zoom_level
    if original_img is None: # Prevent zooming if no image loaded
        return
    if event.delta > 0:  # Zoom in
        zoom_level *= 1.1
    else:  # Zoom out
        zoom_level /= 1.1
    update_canvas()

def start_pan(event):
    global last_x, last_y
    if original_img is None: # Prevent panning if no image loaded
        return
    last_x, last_y = event.x, event.y

def pan(event):
    global offset_x, offset_y, last_x, last_y
    if original_img is None: # Prevent panning if no image loaded
        return
    dx = event.x - last_x
    dy = event.y - last_y
    offset_x += dx
    offset_y += dy
    last_x, last_y = event.x, event.y
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
    if current_tool == "crop" and original_img is not None:
        is_cropping = True
        crop_start_x, crop_start_y = int((event.x - offset_x) / zoom_level), int((event.y - offset_y) / zoom_level)

def stop_crop(event):
    global is_cropping, crop_end_x, crop_end_y
    if current_tool == "crop" and is_cropping and original_img is not None:
        is_cropping = False
        crop_end_x, crop_end_y = int((event.x - offset_x) / zoom_level), int((event.y - offset_y) / zoom_level)
        crop_image()

def crop_image():
    global original_img, crop_start_x, crop_start_y, crop_end_x, crop_end_y
    if original_img is not None and crop_start_x is not None and crop_start_y is not None and crop_end_x is not None and crop_end_y is not None:
        x1, y1 = min(crop_start_x, crop_end_x), min(crop_start_y, crop_end_y)
        x2, y2 = max(crop_start_x, crop_end_x), max(crop_start_y, crop_end_y)

        # Ensure coordinates are within image bounds
        height, width = original_img.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width, x2)
        y2 = min(height, y2)

        if x2 > x1 and y2 > y1: # Ensure a valid crop region
            original_img = original_img[y1:y2, x1:x2]
            update_canvas()
        else:
            messagebox.showwarning("Crop Error", "Invalid crop selection. Please select a valid region.")


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
                img_to_save = cv2.imread(image_path)
                if img_to_save is None:
                    print(f"Warning: Could not read image {image_path}. Skipping.")
                    continue
                img_to_save = cv2.cvtColor(img_to_save, cv2.COLOR_BGR2RGB)

                # Draw annotations
                for shape in annotations_dict[image_path]:
                    if shape["type"] == "circle":
                        cv2.circle(img_to_save, shape["center"], shape["radius"], shape["color"], shape["thickness"])
                    elif shape["type"] == "rectangle":
                        cv2.rectangle(img_to_save, shape["start"], shape["end"], shape["color"], shape["thickness"])
                    elif shape["type"] == "text":
                        cv2.putText(img_to_save, shape["text"], shape["position"], cv2.FONT_HERSHEY_SIMPLEX, shape["font_scale"], shape["color"], shape["thickness"])

                # Resize the image to reduce file size
                height, width = img_to_save.shape[:2]
                max_dimension = 1024  # Set maximum dimension (width or height) to 1024 pixels
                if height > width:
                    new_height = max_dimension
                    new_width = int(width * (max_dimension / height))
                else:
                    new_width = max_dimension
                    new_height = int(height * (max_dimension / width))

                img_resized = cv2.resize(img_to_save, (new_width, new_height), interpolation=cv2.INTER_AREA)

                # Save the annotated image with compression
                # Using a temporary file to check size before adding to zip
                temp_annotated_path = os.path.join(output_dir, f"temp_annotated_{i}.jpg")
                cv2.imwrite(temp_annotated_path, cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])  # Adjust quality here

                # Check the size of the annotated image
                annotated_size = os.path.getsize(temp_annotated_path)

                # If adding this image exceeds the desired size, stop
                if current_zip_size + annotated_size > desired_size_bytes and i > 0: # Allow at least one image
                    messagebox.showwarning("Size Exceeded", f"Adding '{os.path.basename(image_path)}' would exceed the desired ZIP size. Stopping compression.")
                    os.remove(temp_annotated_path) # Clean up temp file
                    break

                # Add the annotated image to the ZIP file
                zipf.write(temp_annotated_path, os.path.basename(image_path).replace('.', '_annotated.')) # Rename in zip
                current_zip_size += annotated_size
                os.remove(temp_annotated_path) # Clean up temporary file

            if current_zip_size > 0:
                messagebox.showinfo("Success", f"Annotated images saved to:\n{zip_path}\nTotal size: {current_zip_size / (1024 * 1024):.2f} MB")
            else:
                messagebox.showwarning("No Images Zipped", "No images were added to the ZIP file, possibly due to size limits or no images uploaded.")
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
    # canvas.unbind("<Button-3>") # Keep right-click for erase_annotation

    # Bind events based on the selected tool
    if tool in ["circle", "rectangle", "text"]:
        canvas.bind("<ButtonPress-1>", start_draw)
        canvas.bind("<ButtonRelease-1>", stop_draw)
    elif tool == "crop":
        canvas.bind("<ButtonPress-1>", start_crop)
        canvas.bind("<ButtonRelease-1>", stop_crop)
    # Right-click to erase is always active for a better UX
    canvas.bind("<Button-3>", erase_annotation)  # Right-click to erase

# Function to Navigate Between Images
def next_image():
    global current_image_index
    if not image_paths:
        return
    if current_image_index < len(image_paths) - 1:
        current_image_index += 1
    load_current_image()
    update_image_counter()

def prev_image():
    global current_image_index
    if not image_paths:
        return
    if current_image_index > 0:
        current_image_index -= 1
    load_current_image()
    update_image_counter()

# Function to Update Image Counter
def update_image_counter():
    if not image_paths:
        counter_label.config(text="Image 0 of 0")
    else:
        counter_label.config(text=f"Image {current_image_index + 1} of {len(image_paths)}")

# NEW: Function to Delete Current Image
def delete_current_image():
    global image_paths, current_image_index, annotations_dict
    if not image_paths:
        messagebox.showwarning("No Image", "No image to delete.")
        return

    confirm = messagebox.askyesno("Confirm Deletion", "Are you sure you want to delete the current image?")
    if confirm:
        deleted_image_path = image_paths[current_image_index]

        # Remove from image_paths
        del image_paths[current_image_index]
        # Remove from annotations_dict
        if deleted_image_path in annotations_dict:
            del annotations_dict[deleted_image_path]

        # Adjust current_image_index
        if len(image_paths) == 0:
            current_image_index = 0
        elif current_image_index >= len(image_paths):
            current_image_index = len(image_paths) - 1
        
        load_current_image() # Load the new current image or clear canvas
        update_image_counter()
        messagebox.showinfo("Image Deleted", f"Image '{os.path.basename(deleted_image_path)}' has been deleted.")


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

    4. **Edit Annotations**:
        - Click "Clear All Markings" to remove all annotations.
        - Click "Undo Last Annotation" to remove the last annotation.
        - Right-click on an annotation to erase it.
        - Click "Delete Current Image" to remove the image entirely from the list.

    5. **Zoom and Pan**:
        - Use the mouse wheel to zoom in or out.
        - Press and hold the middle mouse button to pan.

    6. **Image Transformations**:
        - Rotate images by 90°, 180°, or 270°.
        - Flip images horizontally or vertically.
        - Use the "Crop" tool to select and crop a region.
        - Apply various image filters (Grayscale, Blur, Sharpen, Edge Detection, etc.).
        - "Reset View" restores the image to its original state.

    7. **Save Annotated Images**:
        - Click "Save All to ZIP".
        - Enter the desired ZIP size (MB) and a name for the ZIP file.
        - The annotated images will be saved in a compressed ZIP file.

    8. **Auto-Detect Damage**:
        - Click the "Auto-Detect Damage" button to automatically identify damages using the integrated model. Existing annotations will be cleared, and new damage annotations will be displayed.

    9. **Compare View**:
        - Click "Compare View" to open a new window and compare two selected images side-by-side by entering their 1-based indices.

    **Tips**:
    - Use zoom and pan to work on detailed areas.
    - Right-click to erase specific annotations.
    """
    messagebox.showinfo("How to Use", guide)

# Function to Open Compare View
def open_compare_view():
    global compare_mode, compare_index_1, compare_index_2

    if not image_paths or len(image_paths) < 2:
        messagebox.showwarning("Compare View", "Please upload at least two images to use compare view.")
        return

    # Ask the user to select two images for comparison
    compare_index_1 = simpledialog.askinteger("Compare View", f"Enter the index of the first image (1-{len(image_paths)}):", minvalue=1, maxvalue=len(image_paths))
    compare_index_2 = simpledialog.askinteger("Compare View", f"Enter the index of the second image (1-{len(image_paths)}):", minvalue=1, maxvalue=len(image_paths))

    if compare_index_1 is None or compare_index_2 is None:
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

    # Resize images to fit the window while maintaining aspect ratio
    max_display_width = 500
    max_display_height = 500

    def get_resized_image(image_data):
        h, w = image_data.shape[:2]
        if h > max_display_height or w > max_display_width:
            scale = min(max_display_width / w, max_display_height / h)
            new_w, new_h = int(w * scale), int(h * scale)
            return cv2.resize(image_data, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return image_data

    img1_resized_data = get_resized_image(img1)
    img2_resized_data = get_resized_image(img2)


    # Convert to PhotoImage
    tk_img1 = ImageTk.PhotoImage(Image.fromarray(img1_resized_data))
    tk_img2 = ImageTk.PhotoImage(Image.fromarray(img2_resized_data))

    # Create canvases for the two images
    canvas1 = tk.Canvas(compare_window, width=max_display_width, height=max_display_height, bg="white")
    canvas1.pack(side=tk.LEFT, padx=10, pady=10, expand=True, fill=tk.BOTH)
    canvas1.create_image(max_display_width/2 - tk_img1.width()/2, max_display_height/2 - tk_img1.height()/2, anchor=tk.NW, image=tk_img1) # Center image

    canvas2 = tk.Canvas(compare_window, width=max_display_width, height=max_display_height, bg="white")
    canvas2.pack(side=tk.RIGHT, padx=10, pady=10, expand=True, fill=tk.BOTH)
    canvas2.create_image(max_display_width/2 - tk_img2.width()/2, max_display_height/2 - tk_img2.height()/2, anchor=tk.NW, image=tk_img2) # Center image


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

# Row 0: Upload, Tools, Color Picker
upload_btn = tk.Button(control_frame, text="Upload Images", command=load_images)
upload_btn.grid(row=0, column=0, padx=5, pady=5)

circle_btn = tk.Button(control_frame, text="Circle", command=lambda: set_tool("circle"))
circle_btn.grid(row=0, column=1, padx=5, pady=5)

rect_btn = tk.Button(control_frame, text="Rectangle", command=lambda: set_tool("rectangle"))
rect_btn.grid(row=0, column=2, padx=5, pady=5)

text_btn = tk.Button(control_frame, text="Text", command=lambda: set_tool("text"))
text_btn.grid(row=0, column=3, padx=5, pady=5)

color_btn = tk.Button(control_frame, text="Choose Color", command=choose_color)
color_btn.grid(row=0, column=4, padx=5, pady=5)

color_label = tk.Label(control_frame, text="    ", bg="#%02x%02x%02x" % current_color, relief=tk.SUNKEN)
color_label.grid(row=0, column=5, padx=5, pady=5)

# Row 1: Thickness, Font Size Sliders
tk.Label(control_frame, text="Thickness:").grid(row=1, column=0, padx=5, pady=5)
thickness_slider = tk.Scale(control_frame, from_=1, to=10, orient=tk.HORIZONTAL, command=set_thickness, width=10, length=120)
thickness_slider.set(thickness)
thickness_slider.grid(row=1, column=1, columnspan=2, padx=5, pady=5)

tk.Label(control_frame, text="Font Size:").grid(row=1, column=3, padx=5, pady=5)
font_size_slider = tk.Scale(control_frame, from_=10, to=50, orient=tk.HORIZONTAL, command=set_font_size, width=10, length=120)
font_size_slider.set(font_size)
font_size_slider.grid(row=1, column=4, columnspan=2, padx=5, pady=5)

# Row 2: Image Counter, Navigation, Clear/Undo
counter_label = tk.Label(control_frame, text="Image 0 of 0")
counter_label.grid(row=2, column=0, columnspan=2, padx=5, pady=5)

prev_btn = tk.Button(control_frame, text="Previous", command=prev_image)
prev_btn.grid(row=2, column=2, padx=5, pady=5)

next_btn = tk.Button(control_frame, text="Next", command=next_image)
next_btn.grid(row=2, column=3, padx=5, pady=5)

clear_btn = tk.Button(control_frame, text="Clear All Markings", command=clear_markings)
clear_btn.grid(row=2, column=4, padx=5, pady=5)

undo_btn = tk.Button(control_frame, text="Undo Last Annotation", command=undo_last_annotation)
undo_btn.grid(row=2, column=5, padx=5, pady=5)

# Row 3: Save All, How to Use
save_all_btn = tk.Button(control_frame, text="Save All to ZIP", command=save_all_to_zip)
save_all_btn.grid(row=3, column=0, columnspan=3, padx=5, pady=5)

how_to_use_btn = tk.Button(control_frame, text="How to Use", command=show_how_to_use)
how_to_use_btn.grid(row=3, column=3, columnspan=3, padx=5, pady=5)

# Row 4: Auto-Detect Damage (and new delete button)
detect_damage_btn = tk.Button(control_frame, text="Auto-Detect Damage", command=detect_damage_and_annotate)
detect_damage_btn.grid(row=4, column=0, columnspan=3, padx=5, pady=5) # Adjusted columnspan for new button

delete_image_btn = tk.Button(control_frame, text="Delete Current Image", command=delete_current_image)
delete_image_btn.grid(row=4, column=3, columnspan=3, padx=5, pady=5) # New button

# Row 5: Rotation
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

crop_btn = tk.Button(control_frame, text="Crop", command=lambda: set_tool("crop"))
crop_btn.grid(row=5, column=5, padx=5, pady=5) # Placed crop here

# Row 6: Filters - part 1
grayscale_btn = tk.Button(control_frame, text="Grayscale", command=lambda: apply_filter("grayscale"))
grayscale_btn.grid(row=6, column=0, padx=5, pady=5)

blur_btn = tk.Button(control_frame, text="Blur", command=lambda: apply_filter("blur"))
blur_btn.grid(row=6, column=1, padx=5, pady=5)

sharpen_btn = tk.Button(control_frame, text="Sharpen", command=lambda: apply_filter("sharpen"))
sharpen_btn.grid(row=6, column=2, padx=5, pady=5)

edge_detection_btn = tk.Button(control_frame, text="Edge Detection", command=lambda: apply_filter("edge_detection"))
edge_detection_btn.grid(row=6, column=3, padx=5, pady=5)

contrast_btn = tk.Button(control_frame, text="Contrast", command=lambda: apply_filter("contrast"))
contrast_btn.grid(row=6, column=4, padx=5, pady=5)

color_threshold_btn = tk.Button(control_frame, text="Color Threshold", command=lambda: apply_filter("color_thresholding"))
color_threshold_btn.grid(row=6, column=5, padx=5, pady=5) # Placed here

# Row 7: Filters - part 2
laplacian_btn = tk.Button(control_frame, text="Laplacian", command=lambda: apply_filter("laplacian"))
laplacian_btn.grid(row=7, column=0, padx=5, pady=5)

thermal_btn = tk.Button(control_frame, text="Thermal", command=lambda: apply_filter("thermal"))
thermal_btn.grid(row=7, column=1, padx=5, pady=5)

high_pass_btn = tk.Button(control_frame, text="High-Pass", command=lambda: apply_filter("high_pass"))
high_pass_btn.grid(row=7, column=2, padx=5, pady=5)

# Row 8: Reset View, Compare View
reset_view_btn = tk.Button(control_frame, text="Reset View", command=reset_view)
reset_view_btn.grid(row=8, column=0, columnspan=3, padx=5, pady=5)

compare_view_btn = tk.Button(control_frame, text="Compare View", command=open_compare_view)
compare_view_btn.grid(row=8, column=3, columnspan=3, padx=5, pady=5)

# Canvas for Image Display
canvas_frame = tk.Frame(root)
canvas_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

canvas = tk.Canvas(canvas_frame, bg="white", highlightbackground="grey", highlightthickness=1)
canvas.pack(fill=tk.BOTH, expand=True)

# Bind Mouse Events
canvas.bind("<MouseWheel>", zoom)
canvas.bind("<ButtonPress-2>", start_pan)  # Middle mouse button press
canvas.bind("<B2-Motion>", pan)  # Middle mouse button drag
canvas.bind("<Button-3>", erase_annotation)  # Right-click to erase (always active)

# Set default tool to circle
set_tool("circle")

# NEW: Load the detection model when the GUI starts
load_detection_model()

# Run GUI
root.mainloop()
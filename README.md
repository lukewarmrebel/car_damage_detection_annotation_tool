# Car Damage Annotation & Compression Tool

## Overview

Welcome to the **Car Damage Annotation & Compression Tool**! This powerful and intuitive desktop application, built with Python's Tkinter and OpenCV, is designed for efficiently annotating car damage in images and then compressing those images into a ZIP archive for easy sharing and storage. Whether you're an insurance professional, a car enthusiast, or involved in automotive repair, this tool streamlines your workflow with its rich set of features and user-friendly interface.

## Key Features

* **Bulk Image Upload:** Effortlessly load multiple images for batch processing.
* **Intuitive Annotation Tools:**
    * Draw **circles** and **rectangles** to highlight damaged areas.
    * Add **text annotations** for detailed descriptions.
    * **Customizable Colors & Thickness:** Choose annotation colors and adjust the thickness of shapes and text.
* **Flexible Editing:**
    * **Undo** the last annotation.
    * **Clear all markings** on the current image.
    * **Erase specific annotations** by right-clicking on them.
* **Image Manipulation:**
    * **Rotate** images (90°, 180°, 270°).
    * **Flip** images horizontally or vertically.
    * **Cropping Tool:** Precisely crop images to focus on specific areas.
* **Advanced Image Filters:** Enhance or analyze images with a variety of built-in filters:
    * Grayscale
    * Blur
    * Sharpen
    * Edge Detection
    * Contrast Adjustment
    * Color Thresholding
    * Laplacian
    * Thermal
    * High-Pass
* **Zoom & Pan:** Navigate detailed images with smooth zoom (mouse wheel) and pan (middle mouse button drag) functionalities.
* **Image Navigation:** Easily move between uploaded images using "Previous" and "Next" buttons.
* **Reset View:** Restore the original state of the current image, reverting all applied filters and transformations.
* **Smart Compression to ZIP:**
    * Compress all annotated images into a single ZIP file.
    * **User-defined ZIP size:** Specify a desired maximum size for the ZIP file (in MB), ensuring manageable file sizes for transfer.
    * **Optimized Image Quality:** Images are resized and compressed with adjustable JPEG quality for a balance between file size and visual fidelity.
* **Compare View:** Open a separate window to compare two selected images side-by-side, ideal for before-and-after analysis.
* **Interactive Guide:** A "How to Use" button provides quick instructions within the application.

## Future Developments

We are continuously working to enhance the Car Damage Annotation & Compression Tool. Future developments may include:

* **More Annotation Types:** Adding polygons, arrows, and freehand drawing tools.
* **Annotation Management:** Features to edit existing annotations (e.g., resizing, moving, changing text).
* **Template-based Annotations:** Predefined annotation sets for common damage types.
* **Cloud Integration:** Option to directly upload processed images to cloud storage services.
* **AI-Powered Assistance:** Exploring the integration of AI for automated damage detection and initial annotation suggestions.
* **Customizable Output Formats:** Allowing users to choose different output image formats and compression settings more granularly.
* **Batch Processing of Filters:** Apply filters to multiple images simultaneously.

## Installation and Setup

To run this tool, you need to have Python installed on your system along with a few key libraries.

### Prerequisites

* Python 3.x (Recommended)

### Required Packages

All necessary packages can be installed using `pip`.

Create a `requirements.txt` file in the same directory as your Python script with the following content:

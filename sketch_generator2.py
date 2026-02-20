# """
# sketch_generator.py
# ===================
# Converts person snapshot to pencil sketch style drawing.

# Uses edge detection + adaptive thresholding + smoothing to create
# a sketch effect similar to the example image.
# """

# import cv2
# import numpy as np


# def generate_sketch(image_path: str, output_path: str) -> bool:
#     """
#     Convert a photo to a pencil sketch drawing.
    
#     Args:
#         image_path: Path to input image (person snapshot)
#         output_path: Where to save the sketch
        
#     Returns:
#         True if successful, False otherwise
#     """
#     try:
#         # Read image
#         img = cv2.imread(image_path)
#         if img is None:
#             return False
        
#         # Convert to grayscale
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
#         # ── METHOD 1: Dodge and burn (classic pencil sketch) ────────────────
#         # Invert the grayscale image
#         inverted = cv2.bitwise_not(gray)
        
#         # Apply Gaussian blur to the inverted image
#         blurred = cv2.GaussianBlur(inverted, (21, 21), 0)
        
#         # Invert the blurred image
#         inverted_blur = cv2.bitwise_not(blurred)
        
#         # Create the sketch by dividing the gray image by the inverted blur
#         sketch = cv2.divide(gray, inverted_blur, scale=256.0)
        
#         # ── METHOD 2: Enhance with edge detection ───────────────────────────
#         # Add Canny edges for sharper lines
#         edges = cv2.Canny(gray, 50, 150)
#         edges_inv = cv2.bitwise_not(edges)   # white background, dark edges

#         # Combine sketch with inverted edges (darken edge areas in sketch)
#         sketch_combined = cv2.bitwise_and(sketch, edges_inv)
        
#         # ── Post-processing ──────────────────────────────────────────────────
#         # Apply adaptive threshold for cleaner lines
#         sketch_thresh = cv2.adaptiveThreshold(
#             sketch_combined,
#             255,
#             cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#             cv2.THRESH_BINARY,
#             11,  # block size
#             2    # constant subtracted
#         )
        
#         # Slight blur to smooth jagged edges
#         final_sketch = cv2.GaussianBlur(sketch_thresh, (3, 3), 0)
        
#         # Save as high-quality JPEG
#         cv2.imwrite(output_path, final_sketch, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
#         return True
        
#     except Exception as e:
#         print(f"Sketch generation error: {e}")
#         return False


# def generate_sketch_with_label(image_path: str, output_path: str, 
#                                person_name: str, position_text: str = None) -> bool:
#     """
#     Generate sketch and optionally add text label overlay.
    
#     Args:
#         image_path: Input image path
#         output_path: Output sketch path
#         person_name: Name to label the sketch with
#         position_text: Optional context (e.g. "Person to the right of User1")
        
#     Returns:
#         True if successful
#     """
#     try:
#         # Generate base sketch
#         success = generate_sketch(image_path, output_path)
#         if not success:
#             return False
        
#         # If label requested, add text overlay
#         if person_name:
#             sketch = cv2.imread(output_path)
#             h, w = sketch.shape[:2]
            
#             # Add white background bar at top for text
#             bar_height = 60
#             sketch_with_bar = np.ones((h + bar_height, w, 3), dtype=np.uint8) * 255
#             sketch_with_bar[bar_height:, :] = cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)
            
#             # Add person name
#             cv2.putText(
#                 sketch_with_bar,
#                 person_name.upper(),
#                 (20, 40),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 1.2,
#                 (0, 0, 0),
#                 2,
#                 cv2.LINE_AA
#             )
            
#             # Add position text if provided
#             if position_text:
#                 cv2.putText(
#                     sketch_with_bar,
#                     position_text,
#                     (20, h + bar_height - 15),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     0.6,
#                     (100, 100, 100),
#                     1,
#                     cv2.LINE_AA
#                 )
            
#             cv2.imwrite(output_path, sketch_with_bar, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
#         return True
        
#     except Exception as e:
#         print(f"Sketch labeling error: {e}")
#         return False
"""
sketch_generator.py
===================
Converts person snapshot to pencil sketch style drawing.

Uses edge detection + adaptive thresholding + smoothing to create
a sketch effect similar to the example image.
"""

import cv2
import numpy as np


def _build_sketch(img_bgr: np.ndarray) -> np.ndarray:
    """
    Core sketch algorithm.  Accepts a BGR image, returns a single-channel
    uint8 sketch (white background, dark pencil lines).
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # ── Dodge-and-burn (classic pencil sketch) ────────────────────────────
    inverted      = cv2.bitwise_not(gray)
    blurred       = cv2.GaussianBlur(inverted, (21, 21), 0)
    inverted_blur = cv2.bitwise_not(blurred)
    sketch        = cv2.divide(gray, inverted_blur, scale=256.0)

    # ── Sharpen with Canny edges ──────────────────────────────────────────
    edges     = cv2.Canny(gray, 50, 150)
    edges_inv = cv2.bitwise_not(edges)          # white bg, dark edges
    sketch    = cv2.bitwise_and(sketch, edges_inv)

    # ── Post-processing ───────────────────────────────────────────────────
    sketch = cv2.adaptiveThreshold(
        sketch, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2
    )
    sketch = cv2.GaussianBlur(sketch, (3, 3), 0)
    return sketch   # single-channel grayscale


def generate_sketch(image_path: str, output_path: str) -> bool:
    """
    Convert a photo to a pencil sketch drawing and save it.

    Args:
        image_path:  Path to input image (person snapshot)
        output_path: Where to save the sketch

    Returns:
        True if successful, False otherwise
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return False

        sketch = _build_sketch(img)
        cv2.imwrite(output_path, sketch, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return True

    except Exception as e:
        print(f"Sketch generation error: {e}")
        return False


def generate_sketch_with_label(image_path: str, output_path: str,
                               person_name: str,
                               position_text: str = None) -> bool:
    """
    Generate a labelled pencil-sketch from a snapshot.

    Builds the sketch in memory (no intermediate file round-trip), converts
    it to a 3-channel BGR canvas, then draws the name and position text on a
    white header bar before saving.

    Args:
        image_path:    Input image path
        output_path:   Output sketch path
        person_name:   Name label written at the top
        position_text: Optional context line at the bottom

    Returns:
        True if successful
    """
    try:
        # ── 1. Read original colour photo ────────────────────────────────
        img = cv2.imread(image_path)
        if img is None:
            print(f"Sketch labeling error: cannot read {image_path}")
            return False

        # ── 2. Build the sketch (single-channel grayscale) ───────────────
        sketch_gray = _build_sketch(img)            # shape: (h, w)  1-ch
        h, w        = sketch_gray.shape[:2]

        # ── 3. Convert to 3-channel BGR so we can draw coloured text ─────
        sketch_bgr  = cv2.cvtColor(sketch_gray, cv2.COLOR_GRAY2BGR)

        # ── 4. Create a white canvas with header bar ─────────────────────
        bar_height       = 60
        canvas           = np.ones((h + bar_height, w, 3), dtype=np.uint8) * 255
        canvas[bar_height:, :] = sketch_bgr         # paste sketch below bar

        # ── 5. Draw person name in header ─────────────────────────────────
        font       = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.2
        thickness  = 2
        (tw, _), _ = cv2.getTextSize(person_name.upper(), font, font_scale, thickness)
        if tw > w - 40:                             # shrink if too wide
            font_scale = font_scale * (w - 40) / tw

        cv2.putText(
            canvas,
            person_name.upper(),
            (20, 42),
            font, font_scale,
            (0, 0, 0), thickness, cv2.LINE_AA
        )

        # ── 6. Draw position text at the bottom of the image ─────────────
        if position_text:
            pos_scale = 0.55
            (ptw, _), _ = cv2.getTextSize(position_text, font, pos_scale, 1)
            if ptw > w - 40:
                max_chars = int(len(position_text) * (w - 40) / ptw)
                position_text = position_text[:max_chars - 1] + "..."

            cv2.putText(
                canvas,
                position_text,
                (20, h + bar_height - 12),
                font, pos_scale,
                (80, 80, 80), 1, cv2.LINE_AA
            )

        # ── 7. Save ───────────────────────────────────────────────────────
        cv2.imwrite(output_path, canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return True

    except Exception as e:
        print(f"Sketch labeling error: {e}")
        return False
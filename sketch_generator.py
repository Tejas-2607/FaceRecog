"""
sketch_generator.py
===================
Converts person snapshot to pencil sketch style drawing.

Uses edge detection + adaptive thresholding + smoothing to create
a sketch effect similar to the example image.
"""

import cv2
import numpy as np


def generate_sketch(image_path: str, output_path: str) -> bool:
    """
    Convert a photo to a pencil sketch drawing.
    
    Args:
        image_path: Path to input image (person snapshot)
        output_path: Where to save the sketch
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            return False
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # ── METHOD 1: Dodge and burn (classic pencil sketch) ────────────────
        # Invert the grayscale image
        inverted = cv2.bitwise_not(gray)
        
        # Apply Gaussian blur to the inverted image
        blurred = cv2.GaussianBlur(inverted, (21, 21), 0)
        
        # Invert the blurred image
        inverted_blur = cv2.bitwise_not(blurred)
        
        # Create the sketch by dividing the gray image by the inverted blur
        sketch = cv2.divide(gray, inverted_blur, scale=256.0)
        
        # ── METHOD 2: Enhance with edge detection ───────────────────────────
        # Add Canny edges for sharper lines
        edges = cv2.Canny(gray, 50, 150)
        edges_inv = cv2.bitwise_not(edges)
        
        # Combine sketch with edges
        sketch_combined = cv2.bitwise_and(sketch, sketch_inv)
        
        # ── Post-processing ──────────────────────────────────────────────────
        # Apply adaptive threshold for cleaner lines
        sketch_thresh = cv2.adaptiveThreshold(
            sketch_combined,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,  # block size
            2    # constant subtracted
        )
        
        # Slight blur to smooth jagged edges
        final_sketch = cv2.GaussianBlur(sketch_thresh, (3, 3), 0)
        
        # Save as high-quality JPEG
        cv2.imwrite(output_path, final_sketch, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        return True
        
    except Exception as e:
        print(f"Sketch generation error: {e}")
        return False


def generate_sketch_with_label(image_path: str, output_path: str, 
                               person_name: str, position_text: str = None) -> bool:
    """
    Generate sketch and optionally add text label overlay.
    
    Args:
        image_path: Input image path
        output_path: Output sketch path
        person_name: Name to label the sketch with
        position_text: Optional context (e.g. "Person to the right of User1")
        
    Returns:
        True if successful
    """
    try:
        # Generate base sketch
        success = generate_sketch(image_path, output_path)
        if not success:
            return False
        
        # If label requested, add text overlay
        if person_name:
            sketch = cv2.imread(output_path)
            h, w = sketch.shape[:2]
            
            # Add white background bar at top for text
            bar_height = 60
            sketch_with_bar = np.ones((h + bar_height, w, 3), dtype=np.uint8) * 255
            sketch_with_bar[bar_height:, :] = cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)
            
            # Add person name
            cv2.putText(
                sketch_with_bar,
                person_name.upper(),
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 0, 0),
                2,
                cv2.LINE_AA
            )
            
            # Add position text if provided
            if position_text:
                cv2.putText(
                    sketch_with_bar,
                    position_text,
                    (20, h + bar_height - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (100, 100, 100),
                    1,
                    cv2.LINE_AA
                )
            
            cv2.imwrite(output_path, sketch_with_bar, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        return True
        
    except Exception as e:
        print(f"Sketch labeling error: {e}")
        return False

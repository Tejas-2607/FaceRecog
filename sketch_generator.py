"""
sketch_generator.py
===================
High-quality pencil sketch generator for face/body snapshots.

Uses the classic dodge-burn method with CLAHE pre-processing and
sharpening for clean, natural pencil sketch output.
"""

import cv2
import numpy as np


def _build_sketch(img_bgr: np.ndarray) -> np.ndarray:
    """
    Core sketch pipeline. Accepts BGR uint8, returns grayscale uint8 sketch.
    Uses dodge-burn method with CLAHE enhancement — same approach as sketch_try.py.
    """
    # ── Enhance contrast with CLAHE for better detail recovery ───────────
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_enh = clahe.apply(gray)

    # ── Dodge-burn pencil sketch ──────────────────────────────────────────
    # Step 1: Invert enhanced grayscale
    invert = cv2.bitwise_not(gray_enh)

    # Step 2: Gaussian blur — larger kernel = smoother strokes
    # Kernel size scales with image width so it works on any resolution
    h, w = gray_enh.shape[:2]
    k = max(21, int(w * 0.04) | 1)   # odd, min 21, ~4% of width
    blur = cv2.GaussianBlur(invert, (k, k), 0)

    # Step 3: Invert the blur
    invert_blur = cv2.bitwise_not(blur)

    # Step 4: Colour-dodge divide
    sketch = cv2.divide(gray_enh, invert_blur, scale=256.0)

    # ── Sharpen to crisp up edges ─────────────────────────────────────────
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]], dtype=np.float32)
    sketch = cv2.filter2D(sketch, -1, kernel)
    sketch = np.clip(sketch, 0, 255).astype(np.uint8)

    return sketch


# ─────────────────────────────────────────────────────────────────────────────
#  Public API  (signatures unchanged — drop-in replacement)
# ─────────────────────────────────────────────────────────────────────────────

def generate_sketch(image_path: str, output_path: str) -> bool:
    """
    Convert a photo to a pencil sketch drawing and save it.

    Args:
        image_path:  Path to input image
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

    Args:
        image_path:    Input image path
        output_path:   Output sketch path
        person_name:   Name label written at the top
        position_text: Optional context line at the bottom

    Returns:
        True if successful
    """
    try:
        # ── Read original photo ───────────────────────────────────────────
        img = cv2.imread(image_path)
        if img is None:
            print(f"Sketch labeling error: cannot read {image_path}")
            return False

        # ── Build sketch in memory ────────────────────────────────────────
        sketch_gray = _build_sketch(img)
        h, w        = sketch_gray.shape[:2]

        # ── Convert to 3-channel for text rendering ───────────────────────
        sketch_bgr = cv2.cvtColor(sketch_gray, cv2.COLOR_GRAY2BGR)

        # ── Canvas: header bar + sketch body ─────────────────────────────
        BAR    = 72
        canvas = np.ones((h + BAR, w, 3), dtype=np.uint8) * 255

        # Thin ruled line at bottom of header
        cv2.line(canvas, (0, BAR - 2), (w, BAR - 2), (180, 180, 180), 1)

        # Paste sketch below bar
        canvas[BAR:, :] = sketch_bgr

        # ── Person name ───────────────────────────────────────────────────
        font     = cv2.FONT_HERSHEY_DUPLEX
        name_str = person_name.upper()
        fs       = 1.1
        thick    = 2
        (tw, _), _ = cv2.getTextSize(name_str, font, fs, thick)
        if tw > w - 40:
            fs = fs * (w - 40) / tw
        cv2.putText(canvas, name_str,
                    (20, BAR - 22),
                    font, fs, (30, 30, 30), thick, cv2.LINE_AA)

        # ── Position text ─────────────────────────────────────────────────
        if position_text:
            pfs = 0.52
            (ptw, _), _ = cv2.getTextSize(position_text, font, pfs, 1)
            if ptw > w - 40:
                max_c = int(len(position_text) * (w - 40) / ptw)
                position_text = position_text[:max_c - 1] + "..."
            cv2.putText(canvas, position_text,
                        (20, h + BAR - 10),
                        font, pfs, (100, 100, 100), 1, cv2.LINE_AA)

        # ── Save ──────────────────────────────────────────────────────────
        cv2.imwrite(output_path, canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return True

    except Exception as e:
        print(f"Sketch labeling error: {e}")
        return False
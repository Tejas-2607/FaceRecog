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
    # Slightly larger (5% vs 4%) for bolder strokes suited to laser engraving (change 15)
    h, w = gray_enh.shape[:2]
    k = max(25, int(w * 0.05) | 1)   # odd, min 25, ~5% of width
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


def generate_sketch_for_laser(image_path: str, output_path: str,
                               person_name: str,
                               company: str = "AABBCC") -> bool:
    """
    Generate a laser-engraving-ready sketch with the standard engraving
    text baked in: "Thanking {person_name} — from {company}"

    Identical layout to generate_sketch_with_label() but:
      - Larger blur kernel for stronger engraving strokes (change 15)
      - Header bar contains the person name
      - Footer bar contains the engraving text in the exact laser format
      - Both bars are white-on-dark for easy reading on the board

    Args:
        image_path:  Path to input snapshot
        output_path: Where to save the laser-ready sketch image
        person_name: Name shown in the header (e.g. "Mr. Mohan")
        company:     Company name used in engraving text (default "AABBCC")

    Returns:
        True if successful, False otherwise
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"generate_sketch_for_laser: cannot read {image_path}")
            return False

        # ── Build sketch ──────────────────────────────────────────────────
        sketch_gray = _build_sketch(img)
        h, w        = sketch_gray.shape[:2]
        sketch_bgr  = cv2.cvtColor(sketch_gray, cv2.COLOR_GRAY2BGR)

        # ── Canvas: dark header + sketch body + dark footer ───────────────
        HEADER_H = 60
        FOOTER_H = 52
        canvas   = np.ones((HEADER_H + h + FOOTER_H, w, 3), dtype=np.uint8) * 255

        # Dark header bar
        cv2.rectangle(canvas, (0, 0), (w, HEADER_H), (30, 30, 30), -1)
        # Dark footer bar
        cv2.rectangle(canvas, (0, HEADER_H + h), (w, HEADER_H + h + FOOTER_H),
                      (30, 30, 30), -1)

        # Paste sketch between them
        canvas[HEADER_H: HEADER_H + h, :] = sketch_bgr

        font = cv2.FONT_HERSHEY_DUPLEX

        # ── Header: person name in white ──────────────────────────────────
        name_str = person_name.upper()
        nfs      = 1.0
        nthick   = 2
        (ntw, _), _ = cv2.getTextSize(name_str, font, nfs, nthick)
        if ntw > w - 40:
            nfs = nfs * (w - 40) / ntw
        cv2.putText(canvas, name_str,
                    (20, HEADER_H - 18),
                    font, nfs, (255, 255, 255), nthick, cv2.LINE_AA)

        # ── Footer: engraving text in white ───────────────────────────────
        engrave_text = f"Thanking {person_name}  \u2014  from {company}"
        efs  = 0.6
        (etw, _), _ = cv2.getTextSize(engrave_text, font, efs, 1)
        if etw > w - 40:
            efs = efs * (w - 40) / etw
        text_y = HEADER_H + h + int(FOOTER_H * 0.65)
        cv2.putText(canvas, engrave_text,
                    (20, text_y),
                    font, efs, (255, 255, 255), 1, cv2.LINE_AA)

        # ── Save ──────────────────────────────────────────────────────────
        cv2.imwrite(output_path, canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return True

    except Exception as e:
        print(f"generate_sketch_for_laser error: {e}")
        return False
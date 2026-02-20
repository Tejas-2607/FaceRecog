"""
sketch_generator.py
===================
High-quality pencil sketch generator for face snapshots.

Pipeline:
  1. CLAHE pre-processing  — lifts shadows, reveals detail
  2. Dodge-and-burn base   — classic pencil-sketch luminosity layer
  3. Multi-scale edge map  — fine + coarse Canny blended for natural strokes
  4. Texture overlay       — subtle paper grain so it reads as a real sketch
  5. Tone curve            — darkens midtones, keeps highlights bright white
  6. Label bar             — name + position in a clean ruled header
"""

import cv2
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
#  Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _clahe(gray: np.ndarray) -> np.ndarray:
    """Contrast-limited adaptive histogram equalisation — recovers shadow detail."""
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    return clahe.apply(gray)


def _dodge_burn(gray: np.ndarray, blur_ksize: int = 21) -> np.ndarray:
    """
    Classic colour-dodge pencil-sketch layer.
    Returns a float32 image in [0, 1].
    """
    inv     = 255 - gray
    blurred = cv2.GaussianBlur(inv, (blur_ksize | 1, blur_ksize | 1), 0)
    # colour-dodge: sketch = gray / (1 - blurred/255), clamped
    blur_f  = blurred.astype(np.float32) / 255.0
    gray_f  = gray.astype(np.float32)    / 255.0
    denom   = 1.0 - blur_f
    denom   = np.where(denom < 1e-4, 1e-4, denom)   # avoid /0
    sketch  = np.clip(gray_f / denom, 0.0, 1.0)
    return sketch   # float32 [0,1]


def _multiscale_edges(gray: np.ndarray) -> np.ndarray:
    """
    Blend fine and coarse Canny passes + Laplacian for richer stroke variety.
    Returns uint8 edge mask (255 = edge, 0 = background).
    """
    # Fine edges (hair, pores, fine cloth texture)
    fine   = cv2.Canny(gray, 30,  90)
    # Medium edges (facial features, shoulders)
    medium = cv2.Canny(gray, 60,  160)
    # Coarse structure (head outline, clothing blocks)
    coarse = cv2.Canny(gray, 100, 220)

    # Laplacian catches smooth gradient boundaries Canny misses
    lap    = cv2.Laplacian(gray, cv2.CV_16S, ksize=3)
    lap_u  = cv2.convertScaleAbs(lap)
    _, lap_thresh = cv2.threshold(lap_u, 12, 255, cv2.THRESH_BINARY)

    # Weighted blend: emphasise medium + coarse, sprinkle fine
    edges  = (fine.astype(np.float32)   * 0.25 +
              medium.astype(np.float32) * 0.45 +
              coarse.astype(np.float32) * 0.20 +
              lap_thresh.astype(np.float32) * 0.10)
    edges  = np.clip(edges, 0, 255).astype(np.uint8)

    # Slight dilation so thin strokes look drawn, not printed
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    edges  = cv2.dilate(edges, kernel, iterations=1)
    return edges


def _paper_texture(h: int, w: int, seed: int = 42) -> np.ndarray:
    """
    Generate a subtle fibrous paper-grain texture (float32 in [-0.04, +0.04]).
    Uses only numpy so no extra dependency is needed.
    """
    rng     = np.random.default_rng(seed)
    noise   = rng.standard_normal((h, w)).astype(np.float32) * 0.025
    # Low-pass filter to make it look like paper fibres rather than digital noise
    noise   = cv2.GaussianBlur(noise, (5, 5), 1.2)
    return noise


def _tone_curve(img_f: np.ndarray, gamma: float = 0.82) -> np.ndarray:
    """
    Apply a gentle S-curve:
      - Gamma < 1 darkens midtones (makes strokes heavier, more pencil-like)
      - Bright highlights stay near 1 (white paper)
    Returns float32 [0, 1].
    """
    # Simple power curve; clip keeps it clean
    out = np.power(np.clip(img_f, 0, 1), gamma)
    # Boost near-white region so paper stays crisp
    out = np.where(out > 0.88, 1.0 - (1.0 - out) * 0.4, out)
    return np.clip(out, 0, 1)


def _build_sketch(img_bgr: np.ndarray) -> np.ndarray:
    """
    Full pipeline.  Accepts BGR uint8, returns grayscale uint8 sketch.
    All intermediate work is float32 to avoid quantisation artefacts.
    """
    # ── 0. Upscale small images so details survive ────────────────────────
    h0, w0 = img_bgr.shape[:2]
    scale  = 1
    if max(h0, w0) < 400:
        scale = 2
        img_bgr = cv2.resize(img_bgr, (w0 * scale, h0 * scale),
                             interpolation=cv2.INTER_CUBIC)

    h, w = img_bgr.shape[:2]

    # ── 1. Pre-process ────────────────────────────────────────────────────
    gray      = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray_enh  = _clahe(gray)                        # enhanced for edge detection
    gray_soft = cv2.bilateralFilter(gray, 9, 75, 75)  # smoothed for dodge-burn

    # ── 2. Dodge-and-burn base layer ──────────────────────────────────────
    # Run twice at different blur radii and blend → varies stroke weight
    db_fine   = _dodge_burn(gray_soft, blur_ksize=15)
    db_coarse = _dodge_burn(gray_soft, blur_ksize=31)
    db        = db_fine * 0.55 + db_coarse * 0.45    # float32 [0,1]

    # ── 3. Multi-scale edge layer ─────────────────────────────────────────
    edges     = _multiscale_edges(gray_enh)           # uint8 edge mask
    # Convert: edges are DARK strokes on WHITE paper → invert to [0,1]
    edge_f    = 1.0 - (edges.astype(np.float32) / 255.0)

    # ── 4. Combine: multiply dodge-burn by edge mask ──────────────────────
    # Multiply keeps strokes dark where edges exist AND where shading is dark
    combined  = db * edge_f                           # float32 [0,1]

    # ── 5. Paper texture ──────────────────────────────────────────────────
    texture   = _paper_texture(h, w)
    combined  = np.clip(combined + texture, 0, 1)

    # ── 6. Tone curve — heavier pencil feel ──────────────────────────────
    combined  = _tone_curve(combined, gamma=0.78)

    # ── 7. Final sharpening pass ──────────────────────────────────────────
    combined_u8 = (combined * 255).astype(np.uint8)
    sharp_k     = np.array([[0, -0.5, 0],
                             [-0.5, 3,  -0.5],
                             [0, -0.5, 0]], dtype=np.float32)
    sharpened   = cv2.filter2D(combined_u8, -1, sharp_k)
    sharpened   = np.clip(sharpened, 0, 255).astype(np.uint8)

    # ── 8. Downscale back if we upscaled ─────────────────────────────────
    if scale > 1:
        sharpened = cv2.resize(sharpened, (w0, h0),
                               interpolation=cv2.INTER_AREA)

    return sharpened   # uint8 grayscale


# ─────────────────────────────────────────────────────────────────────────────
#  Public API  (signatures unchanged — drop-in replacement)
# ─────────────────────────────────────────────────────────────────────────────

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

        # ── Build sketch in memory (single-channel) ───────────────────────
        sketch_gray = _build_sketch(img)
        h, w        = sketch_gray.shape[:2]

        # ── Convert to 3-channel for colour text rendering ────────────────
        sketch_bgr  = cv2.cvtColor(sketch_gray, cv2.COLOR_GRAY2BGR)

        # ── Build canvas: ruled header bar + sketch body ──────────────────
        BAR         = 72                                     # header height px
        canvas      = np.ones((h + BAR, w, 3), dtype=np.uint8) * 255

        # Thin ruled line at bottom of header bar
        cv2.line(canvas, (0, BAR - 2), (w, BAR - 2), (180, 180, 180), 1)

        # Paste sketch below bar
        canvas[BAR:, :] = sketch_bgr

        # ── Person name ───────────────────────────────────────────────────
        font        = cv2.FONT_HERSHEY_DUPLEX          # nicer than SIMPLEX
        name_str    = person_name.upper()
        fs          = 1.1
        thick       = 2
        (tw, th), _ = cv2.getTextSize(name_str, font, fs, thick)
        if tw > w - 40:
            fs = fs * (w - 40) / tw
        cv2.putText(canvas, name_str,
                    (20, BAR - 22),
                    font, fs, (30, 30, 30), thick, cv2.LINE_AA)

        # ── Position text (bottom of canvas) ─────────────────────────────
        if position_text:
            pfs         = 0.52
            (ptw, _), _ = cv2.getTextSize(position_text, font, pfs, 1)
            if ptw > w - 40:
                max_c        = int(len(position_text) * (w - 40) / ptw)
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
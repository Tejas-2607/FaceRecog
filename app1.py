from flask import Flask, render_template, Response, request, jsonify, send_file
import base64
import shutil
import cv2
import insightface
import pickle
import numpy as np
from command_parsing_enhanced import CommandParser
from sketch_generator_new import generate_sketch_with_label
import os
from datetime import datetime
import threading
import warnings
import time
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor, as_completed

# pyttsx3 availability check — engine is NOT created here.
# On Windows, pyttsx3 uses COM which is thread-bound: an engine created on
# the main thread silently fails when used from any other thread.
# Fix: create a fresh engine inside each speak() call on the worker thread itself.
try:
    import pyttsx3 as _pyttsx3
    _TTS_AVAILABLE = True
    print("✓ pyttsx3 available — voice output enabled")
except ImportError:
    _pyttsx3      = None
    _TTS_AVAILABLE = False
    print("⚠ pyttsx3 not found — install: pip install pyttsx3 --break-system-packages")
    print("  Voice lines will print to terminal instead.")

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

app = Flask(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
CAMERA_INDEXES         = [0, 1, 2, 3, 4, 5, 6]
DATASET_PATH           = "dataset"
EMBEDDINGS_PATH        = "embeddings/face_embeddings.pkl"
SNAPSHOTS_PATH         = "snapshots"

RECOGNITION_THRESHOLD  = 0.45

# ── Detection size strategy ───────────────────────────────────────────────────
# buffalo_l runs its detector on a downsampled frame. Using (160,160) for the
# live detection loop cuts inference time by ~55% vs (320,320) with only a
# small drop in detecting very small/distant faces (irrelevant at CCTV range).
# The full (320,320) size is used ONLY when taking the final snapshot crop,
# where accuracy matters more than speed.
DETECTION_SIZE         = (160, 160)   # live loop — speed priority
SNAPSHOT_DET_SIZE      = (320, 320)   # snapshot moment — accuracy priority

# ── Frame skip: detect every N frames ────────────────────────────────────────
# Between detection frames, the last known bounding boxes are reused (tracker).
# Raising N from 5→8 means buffalo_l inference runs 37% less often.
# Smooth bounding boxes between frames are handled by the lightweight tracker.
DETECTION_EVERY_N      = 3           # buffalo_l: run every 8th frame, track the rest

# ── Identity cache: skip re-inferring unchanged faces ────────────────────────
# If a face bbox overlaps >85% with a bbox from the previous detection frame,
# reuse the cached name/score instead of re-running the 512-D cosine search.
# Saves the entire embedding + similarity step for static faces.
IDENTITY_CACHE_IOU_THRESH = 0.85     # bbox overlap threshold to reuse cached identity

STREAM_JPEG_QUALITY    = 60
MAX_CONSECUTIVE_ERRORS = 10
FRAME_VALIDATION_ENABLED = True

# ── Input frame downscale for detection ──────────────────────────────────────
# Detect on a smaller frame, draw boxes at original scale.
# 480x270 = 43% fewer pixels than 640x480 → proportionally less memcpy work.
DETECT_FRAME_SCALE     = 0.75        # scale factor applied before passing to InsightFace

# ── Embedding generation config ───────────────────────────────────────────────
EMBEDDING_CACHE_PATH   = "embeddings/file_hash_cache.pkl"  # per-file mtime cache
EMBED_WORKERS          = 2      # keep low on CPU-only laptops to avoid freezing during embedding
EMBED_DET_SIZE         = (160, 160)   # smaller = faster for offline embedding
MIN_FACE_DET_SCORE     = 0.50         # skip very-low-confidence detections
USE_FLIP_AUGMENT       = True         # double data per image (mirror)
AGGREGATE_PER_PERSON   = True         # store 1 mean embedding per person

# ── Camera centering / pan-hint config ───────────────────────────────────────
CENTER_TOLERANCE       = 0.12   # fraction of frame width — dead zone around centre
PAN_HINT_DEGREES       = [10, 15, 20, 30]  # rotation suggestions when no person found

os.makedirs(DATASET_PATH, exist_ok=True)
os.makedirs("embeddings", exist_ok=True)
os.makedirs(SNAPSHOTS_PATH, exist_ok=True)


# ── Fast cosine similarity — replaces sklearn ─────────────────────────────────
def fast_cosine_batch(query_vecs, db_vecs_normalized):
    """
    query_vecs:           (N, D) float32 — normalized here
    db_vecs_normalized:   (M, D) float32 — pre-normalized at load time
    returns:              (N, M) similarity matrix

    Single np.dot (BLAS dgemm) — ~4x faster than sklearn for N < 10 faces
    """
    norms = np.linalg.norm(query_vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    return (query_vecs / norms) @ db_vecs_normalized.T


# ── Bounding-box IoU — used by identity cache ────────────────────────────────
def bbox_iou(a, b):
    """
    Compute Intersection-over-Union between two bboxes [x1,y1,x2,y2].
    Pure NumPy, ~20 ops — negligible cost compared to embedding inference.
    """
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2]-a[0]) * (a[3]-a[1])
    area_b = (b[2]-b[0]) * (b[3]-b[1])
    return inter / (area_a + area_b - inter)


# ── Lightweight bounding-box tracker ─────────────────────────────────────────
class BBoxTracker:
    """
    Dead-simple IoU tracker — no OpenCV contrib needed.

    On every FULL detection frame: update() with fresh InsightFace results.
    On SKIP frames: get_tracked() returns the last known boxes, optionally
    smoothed with a simple exponential average so they don't jump.

    This gives smooth on-screen bounding boxes between buffalo_l inference
    calls (every DETECTION_EVERY_N frames) at near-zero CPU cost.
    """
    SMOOTH = 0.6   # EMA weight for new position (0=fully old, 1=fully new)

    def __init__(self):
        # List of {bbox, name, score, center_x} — last confirmed detections
        self._tracks = []

    def update(self, detected_faces: list):
        """Called every time InsightFace returns a fresh detection result."""
        self._tracks = [dict(d) for d in detected_faces]

    def get_tracked(self) -> list:
        """Return the last known detections (used on skip frames)."""
        return self._tracks

    def smooth_update(self, detected_faces: list):
        """
        Smooth bbox positions using EMA so boxes glide rather than jump.
        Matches new detections to old tracks by IoU.
        """
        if not self._tracks:
            self.update(detected_faces)
            return
        new_tracks = []
        used = set()
        for d in detected_faces:
            best_iou, best_idx = 0.0, -1
            for i, t in enumerate(self._tracks):
                if i in used:
                    continue
                iou = bbox_iou(d["bbox"], t["bbox"])
                if iou > best_iou:
                    best_iou, best_idx = iou, i
            if best_iou > 0.3 and best_idx >= 0:
                used.add(best_idx)
                old = self._tracks[best_idx]["bbox"]
                nb  = d["bbox"]
                # EMA on each coordinate
                sb = np.array([
                    int(self.SMOOTH*nb[0] + (1-self.SMOOTH)*old[0]),
                    int(self.SMOOTH*nb[1] + (1-self.SMOOTH)*old[1]),
                    int(self.SMOOTH*nb[2] + (1-self.SMOOTH)*old[2]),
                    int(self.SMOOTH*nb[3] + (1-self.SMOOTH)*old[3]),
                ], dtype=np.int32)
                new_tracks.append({**d, "bbox": sb,
                                   "center_x": (sb[0]+sb[2])>>1})
            else:
                new_tracks.append(d)
        self._tracks = new_tracks


# ── Lock-free single-slot buffer ──────────────────────────────────────────────
class AtomicFrameSlot:
    """
    Writer always overwrites with latest value.
    Reader always gets latest value.
    Neither ever blocks — a single lightweight mutex guards only the pointer swap.
    """
    def __init__(self):
        self._lock  = threading.Lock()
        self._frame = None
        self._meta  = None

    def write(self, frame, meta=None):
        with self._lock:
            self._frame = frame
            self._meta  = meta

    def read(self):
        with self._lock:
            return self._frame, self._meta


class EncodedFrameSlot:
    """Stores the latest pre-encoded JPEG bytes."""
    def __init__(self):
        self._lock  = threading.Lock()
        self._bytes = None

    def write(self, data: bytes):
        with self._lock:
            self._bytes = data

    def read(self):
        with self._lock:
            return self._bytes


# ── System state ──────────────────────────────────────────────────────────────
class SystemState:

    def __init__(self):
        self.camera            = None
        self.recognizer        = None
        self.known_embeddings  = None
        self.known_names       = None
        self.command_parser    = CommandParser()
        self.current_command   = None

        # Camera bookkeeping
        self.camera_lock           = threading.Lock()
        self.consecutive_errors    = 0
        self.last_successful_frame = time.time()
        self.camera_url_index      = 0
        self.frame_count           = 0

        # Lock-free frame slots (replace all Lock + frame variable pairs)
        self.raw_slot       = AtomicFrameSlot()   # latest raw frame
        self.annotated_slot = AtomicFrameSlot()   # latest annotated frame
        self.encoded_slot   = EncodedFrameSlot()  # latest JPEG bytes

        # Detection input — maxsize=1, always drop stale frame if detect is busy
        self.detect_queue = Queue(maxsize=1)

        # Worker threads
        self.detect_thread   = None
        self.encode_thread   = None
        self.threads_running = False

        # Latest detection info (for API)
        self.detection_results = {}

        # Normalized embeddings (pre-computed at load)
        self.normalized_embeddings = None

        # ── Snapshot — one-shot script-driven mode ────────────────────────
        # snapshot_locked = True means a photo is already taken and waiting
        # for the operator to finish the verify → sketch → confirm flow.
        # Nothing new is taken until /api/reset_snapshot is called.
        self.auto_snapshot_enabled = True
        self.snapshot_locked       = False
        self.last_snapshot_person  = None
        self._tts_lock             = threading.Lock()
        self.is_speaking           = False   # True while TTS is playing + 500ms tail
        # Pending auto-crop waiting for frontend verification
        self.pending_auto_snapshot = None

        # Pre-allocated draw buffer — reused every frame to avoid repeated ~921KB malloc
        self._draw_buf = None

        # Pan-hint string cache — only rebuild when text actually changes
        self._pan_hint_cache     = None
        self._pan_hint_cache_key = None

        # ── BBox tracker — smooth boxes on skip frames ────────────────────
        self._tracker = BBoxTracker()

        # ── Identity cache — skip re-embedding faces that haven't moved ───
        # Stores {bbox_tuple: (name, score)} from the last full detection.
        # On the next full detection, faces with IoU > IDENTITY_CACHE_IOU_THRESH
        # against a cached entry skip the cosine similarity step entirely.
        self._identity_cache = {}   # {(x1,y1,x2,y2): (name, score)}

        # ── Downscaled detection frame reuse ─────────────────────────────
        # Pre-allocate the small frame buffer to avoid repeated malloc.
        self._detect_small_buf = None
        self._detect_skip_count = 0   # frames skipped since last full detection

    # ── Model init ────────────────────────────────────────────────────────────

    def initialize_recognizer(self):
        if self.recognizer is not None:
            return
        print("Loading InsightFace model...")
        # Try GPU first — explicitly set allowed_modules so ONNX Runtime
        # picks the CUDA EP for EVERY sub-model (det + rec + landmark).
        for providers in (
            ["CUDAExecutionProvider", "CPUExecutionProvider"],
            ["CPUExecutionProvider"],
        ):
            try:
                self.recognizer = insightface.app.FaceAnalysis(
                    name="buffalo_l",
                    providers=providers,
                    allowed_modules=["detection", "recognition"]
                )
                ctx = 0 if "CUDA" in providers[0] else -1
                self.recognizer.prepare(ctx_id=ctx, det_size=DETECTION_SIZE)
                label = "GPU (RTX)" if ctx == 0 else "CPU"
                print(f"✓ InsightFace loaded ({label}) det_size={DETECTION_SIZE}")
                return
            except Exception as e:
                print(f"  {providers[0]} failed: {e}")

    def load_embeddings(self):
        if not os.path.exists(EMBEDDINGS_PATH):
            return False
        try:
            with open(EMBEDDINGS_PATH, "rb") as f:
                data = pickle.load(f)
            emb   = np.array(data["embeddings"], dtype=np.float32)
            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            norms[norms == 0] = 1e-10
            self.normalized_embeddings = emb / norms
            self.known_names           = data["names"]
            print(f"✓ Loaded {len(self.known_names)} faces")
            return True
        except Exception as e:
            print(f"❌ Embeddings error: {e}")
            return False

    # ── Text-to-speech ───────────────────────────────────────────────────────

    def speak(self, text: str):
        """
        Speak text aloud via pyttsx3 — non-blocking, never delays detection.

        Engine is created fresh inside the worker thread on every call.
        This is intentional: pyttsx3 on Windows uses COM which is thread-bound.
        An engine initialised on the main thread silently does nothing when
        runAndWait() is called from a different thread. Creating it on the same
        thread that calls runAndWait() fixes the silence.

        Echo / feedback-loop prevention:
          is_speaking is set True before runAndWait() and kept True for
          MIC_MUTE_TAIL_MS after engine.stop(). The frontend polls
          /api/speaking_state and holds both mic paths closed during this
          entire window so the AI voice never feeds back into the mic.
        """
        MIC_MUTE_TAIL_MS = 500   # ms to keep mic muted after TTS ends
                                 # 500ms covers BT buffer tail; reduce to 250ms
                                 # for wired headsets if responses feel sluggish
        print(f"[SYSTEM] {text}")
        if not _TTS_AVAILABLE:
            return
        def _run():
            with self._tts_lock:
                self.is_speaking = True
                try:
                    engine = _pyttsx3.init()
                    engine.setProperty("rate", 160)
                    engine.setProperty("volume", 1.0)
                    engine.say(text)
                    engine.runAndWait()
                    engine.stop()
                except Exception as e:
                    print(f"TTS error: {e}")
                finally:
                    time.sleep(MIC_MUTE_TAIL_MS / 1000.0)
                    self.is_speaking = False
        threading.Thread(target=_run, daemon=True).start()

    # ── Camera ────────────────────────────────────────────────────────────────

    def _open_camera(self, index):
        cam = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not cam.isOpened():
            cam.release()
            cam = cv2.VideoCapture(index)
        if cam.isOpened():
            cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cam.set(cv2.CAP_PROP_FPS, 30)
            cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        return cam

    def get_camera(self):
        with self.camera_lock:
            if self.camera is not None and self.camera.isOpened():
                if time.time() - self.last_successful_frame < 5.0:
                    return self.camera
                print("⚠ Camera timeout, reconnecting...")
                self.camera.release()
                self.camera = None

            if self.camera is not None:
                self.camera.release()
                self.camera = None

            for _ in range(len(CAMERA_INDEXES)):
                idx = CAMERA_INDEXES[self.camera_url_index]
                print(f"Trying camera {idx}...")
                self.camera = self._open_camera(idx)
                if self.camera.isOpened():
                    print(f"✓ Camera {idx} connected")
                    self.consecutive_errors    = 0
                    self.last_successful_frame = time.time()
                    return self.camera
                self.camera_url_index = (self.camera_url_index + 1) % len(CAMERA_INDEXES)
                print(f"❌ Camera {idx} failed")

            print("❌ All cameras failed")
            return None

    def release_camera(self):
        with self.camera_lock:
            if self.camera is not None:
                self.camera.release()
                self.camera = None

    # ── Frame validation — fast corner sampling ───────────────────────────────

    @staticmethod
    def validate_frame(frame):
        """
        Sample 5 pixels (4 corners + center) instead of computing mean of entire frame.
        640x480x3 full mean = 921,600 operations.
        Corner sampling = 5 operations. ~200x faster.
        """
        if frame is None or frame.size == 0:
            return False
        if not FRAME_VALIDATION_ENABLED:
            return True
        h, w = frame.shape[:2]
        samples = (
            frame[0, 0], frame[0, w-1],
            frame[h-1, 0], frame[h-1, w-1],
            frame[h//2, w//2]
        )
        mean = sum(int(s.mean()) for s in samples) / 5
        return 5 < mean < 250

    # ── Worker thread management ──────────────────────────────────────────────

    def start_threads(self):
        if self.threads_running:
            return
        self.threads_running = True
        self.detect_thread = threading.Thread(
            target=self._detect_worker, daemon=True, name="DetectThread"
        )
        self.encode_thread = threading.Thread(
            target=self._encode_worker, daemon=True, name="EncodeThread"
        )
        self.detect_thread.start()
        self.encode_thread.start()
        print("✓ Detection + Encode threads started")

    # ── Detection worker ──────────────────────────────────────────────────────

    def _detect_worker(self):
        """
        Runs at its own pace — completely decoupled from stream FPS.
        If InsightFace takes 100ms, stream still runs at 30 FPS unaffected.

        Optimization: every DETECTION_EVERY_N frames a full InsightFace inference
        runs. Between those frames, the tracker's last known boxes are used to
        redraw the annotated frame without any inference cost. This keeps the
        displayed bounding boxes smooth at the full camera FPS.
        """
        skip_budget = 0   # frames to serve from tracker before next full inference
        while self.threads_running:
            try:
                frame, command = self.detect_queue.get(timeout=0.05)
            except Empty:
                continue
            try:
                if skip_budget > 0:
                    # ── Tracker frame: reuse last known boxes, skip inference ──
                    skip_budget -= 1
                    tracked = self._tracker.get_tracked()
                    if tracked:
                        # Re-draw with tracker boxes — zero inference cost
                        annotated, info = self._redraw_tracked(frame, tracked, command)
                        self.annotated_slot.write(annotated, info)
                        self.detection_results = info
                    else:
                        # No tracks yet — fall through to full inference
                        skip_budget = 0
                        annotated, info = self._run_detection(frame, command)
                        self.annotated_slot.write(annotated, info)
                        self.detection_results = info
                        skip_budget = DETECTION_EVERY_N - 1
                else:
                    # ── Full inference frame ──────────────────────────────────
                    annotated, info = self._run_detection(frame, command)
                    self.annotated_slot.write(annotated, info)
                    self.detection_results = info
                    skip_budget = DETECTION_EVERY_N - 1
            except Exception as e:
                print(f"Detection error: {e}")
                skip_budget = 0
            self.detect_queue.task_done()

    # ── Encode worker ─────────────────────────────────────────────────────────

    def _encode_worker(self):
        """
        Continuously encodes the annotated frame to JPEG.
        Only re-encodes when the frame object changes (id() check = free).
        Stream thread just copies bytes — zero encode work on hot path.
        """
        params   = [cv2.IMWRITE_JPEG_QUALITY, STREAM_JPEG_QUALITY]
        prev_id  = None

        while self.threads_running:
            annotated, _ = self.annotated_slot.read()
            cur_id = id(annotated) if annotated is not None else None

            if cur_id == prev_id or annotated is None:
                time.sleep(0.002)  # 2ms poll — negligible CPU when idle
                continue

            prev_id = cur_id
            ret, buf = cv2.imencode('.jpg', annotated, params)
            if ret:
                self.encoded_slot.write(buf.tobytes())

    # ── Detection logic ───────────────────────────────────────────────────────

    def _redraw_tracked(self, frame, tracked_faces: list, command_result):
        """
        Redraw bounding boxes using the tracker's last known positions.
        No InsightFace inference — just draws existing boxes on the current frame.
        Cost: O(N) bbox draws instead of full ONNX forward pass.
        """
        anchor_face    = None
        detected_faces = tracked_faces  # already have name/score/bbox/center_x

        # Rebuild anchor from tracked data
        if command_result:
            ref = command_result.get('reference_person')
            for d in detected_faces:
                if d["name"] == ref:
                    anchor_face = d
                    break

        # Determine target (same logic as _run_detection)
        target_detected = False
        target_face     = None
        if command_result and anchor_face is not None:
            direction  = command_result.get('direction')
            wanted_pos = command_result.get('position', 1)
            anchor_cx  = anchor_face["center_x"]
            anchor_name = command_result.get('reference_person')

            if command_result.get('mode') == 'single':
                target_detected = True
                target_face     = anchor_face
            else:
                side_faces = []
                for d in detected_faces:
                    if d["name"] == anchor_name:
                        continue
                    on_side = ((direction == "right" and d["center_x"] < anchor_cx) or
                               (direction == "left"  and d["center_x"] > anchor_cx))
                    if on_side:
                        side_faces.append((abs(d["center_x"] - anchor_cx), d))
                side_faces.sort(key=lambda t: t[0])
                if len(side_faces) >= wanted_pos:
                    _, target_face = side_faces[wanted_pos - 1]
                    target_detected = True

        annotated = self._draw(frame, detected_faces, anchor_face,
                               command_result, target_face, None, None)
        return annotated, {
            'total_faces':     len(detected_faces),
            'anchor_detected': anchor_face is not None if command_result else False,
            'target_detected': target_detected,
            'center_hint':     None,
            'pan_hint':        None,
            'faces': [{'name': f['name'], 'score': float(f['score'])}
                      for f in detected_faces]
        }

    def _run_detection(self, frame, command_result):
        if self.recognizer is None:
            self.initialize_recognizer()
        if self.normalized_embeddings is None:
            self.load_embeddings()

        if self.normalized_embeddings is None or len(self.normalized_embeddings) == 0:
            # No dataset / embeddings — draw live video with a warning overlay.
            # IMPORTANT: must return frame.copy() not _draw_buf so the encode
            # thread gets a NEW object id every call — otherwise id() stays the
            # same across calls and the encode thread thinks the frame hasn't
            # changed, producing a frozen stream.
            out = frame.copy()
            # Semi-transparent dark bar at top so text is always readable
            fh_o, fw_o = out.shape[:2]
            cv2.rectangle(out, (0, 0), (fw_o, 56), (20, 20, 20), -1)
            cv2.putText(out, "No persons in dataset — add users to begin detection",
                        (14, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 180, 255), 2,
                        cv2.LINE_AA)
            return out, {'total_faces': 0, 'message': 'No dataset'}

        fh, fw = frame.shape[:2]

        # ── Downscale frame for InsightFace detection ─────────────────────────
        # Detect on a smaller frame → fewer pixels → faster ONNX inference.
        # Bboxes are scaled back up to original resolution before drawing.
        # DETECT_FRAME_SCALE = 0.75 → 480x360 instead of 640x480 (43% fewer pixels)
        if DETECT_FRAME_SCALE < 1.0:
            small_w = int(fw * DETECT_FRAME_SCALE)
            small_h = int(fh * DETECT_FRAME_SCALE)
            # Reuse pre-allocated buffer if shape matches — avoids malloc
            if (self._detect_small_buf is None or
                    self._detect_small_buf.shape[:2] != (small_h, small_w)):
                self._detect_small_buf = np.empty((small_h, small_w, 3), dtype=np.uint8)
            cv2.resize(frame, (small_w, small_h),
                       dst=self._detect_small_buf,
                       interpolation=cv2.INTER_LINEAR)
            detect_frame = self._detect_small_buf
            scale_x = fw / small_w
            scale_y = fh / small_h
        else:
            detect_frame = frame
            scale_x = scale_y = 1.0

        # Contiguous array avoids InsightFace's internal copy
        faces = self.recognizer.get(np.ascontiguousarray(detect_frame))

        anchor_face    = None
        detected_faces = []

        if faces:
            # ── Identity cache: skip cosine search for stationary faces ────
            # Build list of faces that need full embedding inference vs cached.
            need_inference = []
            cached_results = []
            for face in faces:
                raw_bbox = face.bbox.astype(int)
                # Scale bbox back to original frame coordinates
                bbox = np.array([
                    int(raw_bbox[0] * scale_x), int(raw_bbox[1] * scale_y),
                    int(raw_bbox[2] * scale_x), int(raw_bbox[3] * scale_y),
                ], dtype=np.int32)
                bbox_key = tuple(bbox.tolist())

                # Check identity cache — O(N*M) but N,M < 10 so negligible
                cache_hit = None
                for cached_key, (c_name, c_score) in self._identity_cache.items():
                    if bbox_iou(bbox_key, cached_key) >= IDENTITY_CACHE_IOU_THRESH:
                        cache_hit = (c_name, c_score, bbox)
                        break

                if cache_hit:
                    cached_results.append(cache_hit)
                else:
                    need_inference.append((face, bbox))

            # Run cosine similarity only for faces NOT in cache
            new_cache = {}
            if need_inference:
                emb_batch = np.array([f.embedding for f, _ in need_inference],
                                     dtype=np.float32)
                sims = fast_cosine_batch(emb_batch, self.normalized_embeddings)
                for i, (face, bbox) in enumerate(need_inference):
                    best_idx   = int(np.argmax(sims[i]))
                    best_score = float(sims[i, best_idx])
                    name = (self.known_names[best_idx]
                            if best_score > RECOGNITION_THRESHOLD else "Unknown")
                    bbox_key = tuple(bbox.tolist())
                    new_cache[bbox_key] = (name, best_score)
                    cached_results.append((name, best_score, bbox))

            # Update identity cache with new detections
            self._identity_cache = new_cache

            # Build detected_faces from combined cached + fresh results
            for name, best_score, bbox in cached_results:
                x1, y1, x2, y2 = bbox
                center_x = (x1 + x2) >> 1
                detected_faces.append({
                    "name": name, "score": best_score,
                    "bbox": bbox, "center_x": center_x
                })
                if command_result and name == command_result.get('reference_person'):
                    anchor_face = {"bbox": bbox, "center_x": center_x,
                                   "name": name, "score": best_score}

            # Update tracker with fresh detections for smooth interpolation
            self._tracker.smooth_update(detected_faces)
        else:
            # No faces from InsightFace — clear cache
            self._identity_cache = {}
            self._tracker.update([])

        # ── Camera centering hint (Feature 4) ─────────────────────────────
        # If a command is active and the anchor person is found but NOT centred,
        # tell the operator to pan the camera.
        center_hint = None
        if command_result and anchor_face is not None:
            frame_cx    = fw // 2
            anchor_cx   = anchor_face["center_x"]
            dead_zone   = int(fw * CENTER_TOLERANCE)
            offset      = anchor_cx - frame_cx
            if abs(offset) > dead_zone:
                pan_dir    = "LEFT" if offset > 0 else "RIGHT"
                pct        = abs(offset) / fw * 100
                center_hint = f"⟵ Pan camera {pan_dir} to centre {command_result['reference_person']} ({pct:.0f}%)"

        # ── Determine target_detected with POSITION support (Feature 1) ───
        target_detected = False
        target_face     = None
        if command_result and anchor_face is not None:
            # Extract once — avoids repeated .get() calls throughout this block
            direction      = command_result.get('direction')
            wanted_pos     = command_result.get('position', 1)  # 1-based
            anchor_cx      = anchor_face["center_x"]
            anchor_name    = command_result.get('reference_person')
            ref_person     = command_result.get('mode')

            if ref_person == 'single':
                # Single-person mode: just needs anchor in frame
                target_detected = True
                target_face     = anchor_face
            else:
                # Directional mode — collect all qualifying faces sorted by
                # proximity to anchor, then pick the Nth one.
                side_faces = []
                for d in detected_faces:
                    if d["name"] == anchor_name:
                        continue
                    on_side = ((direction == "right" and d["center_x"] < anchor_cx) or
                               (direction == "left"  and d["center_x"] > anchor_cx))
                    if on_side:
                        # Distance from anchor determines 1st/2nd/3rd ordering
                        dist = abs(d["center_x"] - anchor_cx)
                        side_faces.append((dist, d))

                # Sort ascending by distance — closest face to the anchor is always
                # position 1, regardless of direction. The on_side filter above already
                # ensures only faces on the correct side are included.
                # Bug fix: previously used reverse=(direction=="right") which incorrectly
                # placed the furthest face at position 1 when direction is "right".
                side_faces.sort(key=lambda t: t[0])  # ascending distance, no reverse

                if len(side_faces) >= wanted_pos:
                    _, target_face = side_faces[wanted_pos - 1]
                    target_detected = True

        # ── Pan hint when no person found on commanded side (Feature 5) ───
        pan_hint = None
        if (command_result and anchor_face is not None
                and command_result.get('mode') == 'directional'
                and not target_detected):
            direction = command_result.get('direction', '')
            wanted_pos = command_result.get('position', 1)
            side_count = len([d for d in detected_faces
                               if d["name"] != command_result.get('reference_person')])
            if side_count == 0:
                hints = ", ".join([f"{d}°" for d in PAN_HINT_DEGREES])
                _new_hint = (f"↔ No person found — rotate camera {direction.upper()} "
                             f"by {hints} to search")
            else:
                _new_hint = (f"↔ Only {side_count} person(s) visible — try rotating "
                             f"camera {direction.upper()} to find person #{wanted_pos}")
            # Only rebuild string object when content actually changes
            _cache_key = (direction, side_count, wanted_pos)
            if _cache_key != self._pan_hint_cache_key:
                self._pan_hint_cache     = _new_hint
                self._pan_hint_cache_key = _cache_key
            pan_hint = self._pan_hint_cache

        # Auto-snapshot on clean frame (before drawing boxes)
        if command_result and target_face is not None and self.auto_snapshot_enabled:
            self._check_snapshot_targeted(frame, target_face, command_result)

        annotated = self._draw(frame, detected_faces, anchor_face,
                               command_result, target_face, center_hint, pan_hint)
        return annotated, {
            'total_faces':     len(detected_faces),
            'anchor_detected': anchor_face is not None if command_result else False,
            'target_detected': target_detected,
            'center_hint':     center_hint,
            'pan_hint':        pan_hint,
            'faces': [{'name': f['name'], 'score': float(f['score'])}
                      for f in detected_faces]
        }

    def _check_snapshot_targeted(self, frame, target_face, command_result):
        """
        One-shot snapshot matching the demo script flow:
          1. Target detected for the first time → take one photo, lock.
          2. System speaks confirmation question.
          3. Operator works through the 3-step modal (person → snapshot → sketch).
          4. Any Retake/Discard calls /api/reset_snapshot → unlocks for next shot.
        No more automatic 8-second repeat.
        """
        if self.snapshot_locked:
            return
        name = target_face.get("name", "Unknown")
        if self.last_snapshot_person == name:
            return

        # Lock immediately so concurrent frames don't double-fire
        self.snapshot_locked      = True
        self.last_snapshot_person = name
        self._save_crop(frame, target_face["bbox"], name, command_result)
        # NOTE: Do NOT call self.speak() here — the frontend speaks the
        # confirmation question exactly once when it opens the modal.
        # Calling speak() here caused the question to be said twice.

    def _save_crop(self, frame, bbox, person_name, command_result=None):
        try:
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = bbox

            # ── Full-body crop ─────────────────────────────────────────────
            face_h  = y2 - y1
            face_w  = x2 - x1
            face_cx = (x1 + x2) // 2

            body_height = int(face_h * 7.0)
            body_width  = int(face_w * 4.5)
            top_margin  = int(face_h * 0.35)

            crop_x1 = max(0, face_cx - body_width  // 2)
            crop_x2 = min(w, face_cx + body_width  // 2)
            crop_y1 = max(0, y1      - top_margin)
            crop_y2 = min(h, y1      + body_height)

            crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
            if crop.size == 0:
                return
            ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
            fn  = f"auto_{person_name.replace(' ','_')}_{ts}.jpg"
            cv2.imwrite(os.path.join(SNAPSHOTS_PATH, fn), crop,
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            print(f"📸 {fn}")

            # Build position description for verification UI
            pos_desc = ''
            if command_result:
                mode = command_result.get('mode', 'directional')
                ref  = command_result.get('reference_person', '')
                pos  = command_result.get('position', 1)
                if mode == 'single':
                    pos_desc = f"Detected: {person_name}"
                else:
                    ordinal = {1:'1st',2:'2nd',3:'3rd'}.get(pos, f'{pos}th')
                    pos_desc = (f"{ordinal} person to the "
                                f"{command_result.get('direction','')} of {ref}")

            self.pending_auto_snapshot = {
                'filename':     fn,
                'person_name':  person_name,
                'position_desc': pos_desc,
                'timestamp':    datetime.now().isoformat()
            }
        except Exception as e:
            print(f"Snapshot error: {e}")

    def _draw(self, frame, detected_faces, anchor_face, command_result,
              target_face=None, center_hint=None, pan_hint=None):
        # _draw_buf is a scratch buffer for drawing — we copy into it, draw on it,
        # then return a SEPARATE copy so the encode thread never touches a buffer
        # the detect thread is still writing into (freeze root cause on CPU).
        if self._draw_buf is None or self._draw_buf.shape != frame.shape:
            self._draw_buf = np.empty_like(frame)
        np.copyto(self._draw_buf, frame)
        frame = self._draw_buf
        fh, fw = frame.shape[:2]
        anchor_name = command_result.get('reference_person') if command_result else None
        direction   = command_result.get('direction') if command_result else None
        wanted_pos  = command_result.get('position', 1) if command_result else 1
        mode        = command_result.get('mode') if command_result else None

        # ── Draw all non-anchor, non-target faces (blue) ─────────────────
        target_bbox = target_face["bbox"] if target_face else None
        for d in detected_faces:
            x1, y1, x2, y2 = d["bbox"]
            is_anchor  = (d["name"] == anchor_name)
            is_target  = (target_bbox is not None and
                          list(d["bbox"]) == list(target_bbox))
            if is_anchor or is_target:
                continue
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 80, 0), 2)
            cv2.putText(frame, f"{d['name']} ({d['score']:.2f})",
                        (x1, max(y1-10, 12)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 80, 0), 2)

        # ── Draw anchor face (green) ──────────────────────────────────────
        if anchor_face is not None:
            ax1, ay1, ax2, ay2 = anchor_face["bbox"]
            cv2.rectangle(frame, (ax1, ay1), (ax2, ay2), (0, 220, 0), 3)
            cv2.putText(frame, f"{anchor_name} [Anchor]",
                        (ax1, max(ay1-12, 12)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 220, 0), 2)

        # ── Draw target face (cyan/magenta) ───────────────────────────────
        if target_face is not None:
            tx1, ty1, tx2, ty2 = target_face["bbox"]
            ordinal = {1:'1st',2:'2nd',3:'3rd'}.get(wanted_pos, f'{wanted_pos}th')
            label   = f"✓ TARGET ({ordinal}) {target_face['name']} ({target_face['score']:.2f})"
            cv2.rectangle(frame, (tx1, ty1), (tx2, ty2), (255, 0, 255), 3)
            cv2.putText(frame, label,
                        (tx1, max(ty1-12, 12)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (255, 0, 255), 2)

        # ── Status line at top-left ───────────────────────────────────────
        y_cursor = 38
        if command_result and anchor_face is None:
            cv2.putText(frame, f"❌ {anchor_name} not in frame", (20, y_cursor),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            y_cursor += 44
        elif command_result and target_face is not None:
            ordinal = {1:'1st',2:'2nd',3:'3rd'}.get(wanted_pos, f'{wanted_pos}th')
            cv2.putText(frame, f"✓ {ordinal} person {direction} of {anchor_name}: {target_face['name']}",
                        (20, y_cursor), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 255, 0), 2)
            y_cursor += 44
        elif command_result and anchor_face is not None and not target_face:
            ordinal = {1:'1st',2:'2nd',3:'3rd'}.get(wanted_pos, f'{wanted_pos}th')
            cv2.putText(frame, f"❌ No {ordinal} person found {direction} of {anchor_name}",
                        (20, y_cursor), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 0, 255), 2)
            y_cursor += 44

        # ── Centre hint (feature 4) ───────────────────────────────────────
        if center_hint:
            # Draw arrow pointing in pan direction
            cv2.putText(frame, center_hint, (20, y_cursor),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 200, 255), 2)
            y_cursor += 36

        # ── Pan hint (feature 5) — shown at bottom of frame ──────────────
        if pan_hint:
            # Direct filled rectangle — no frame.copy() or addWeighted needed
            cv2.rectangle(frame, (0, fh - 52), (fw, fh), (20, 20, 60), -1)
            cv2.putText(frame, pan_hint, (20, fh - 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 220, 255), 2)

        # Return a copy — encode thread must get its own buffer, not _draw_buf
        # which the detect thread will overwrite on the next frame.
        return frame.copy()


state = SystemState()


# ============================================================================
# STREAM — pure hot path, zero blocking operations
# ============================================================================

def generate_frames():
    """
    Stream loop does ONLY:
      1. camera.grab()       — flush buffer, get latest frame
      2. camera.retrieve()   — decode frame
      3. cv2.flip()          — mirror
      4. slot.write()        — atomic pointer swap (nanoseconds)
      5. detect_queue.put_nowait() — non-blocking, drops if busy
      6. encoded_slot.read() — atomic pointer read (nanoseconds)
      7. yield bytes          — send to browser

    No JPEG encoding. No face detection. No lock contention.
    """
    camera = state.get_camera()
    if camera is None:
        err = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(err, "Camera Error", (50, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        _, buf = cv2.imencode('.jpg', err)
        yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n'
        return

    state.start_threads()

    # Pre-build header/footer bytes — avoids any string/bytes ops per frame
    HEADER = b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
    FOOTER = b'\r\n'

    # Fallback encoder only used for very first frames before encode thread warms up
    fallback_params = [cv2.IMWRITE_JPEG_QUALITY, STREAM_JPEG_QUALITY]
    error_cache = None

    while True:
        try:
            # grab() flushes camera buffer and returns immediately if frame ready
            if not camera.grab():
                state.consecutive_errors += 1
                if state.consecutive_errors > MAX_CONSECUTIVE_ERRORS:
                    print("⚠ Reconnecting...")
                    camera = state.get_camera()
                    if camera is None:
                        time.sleep(0.5)
                continue

            ret, frame = camera.retrieve()

            if not ret or not state.validate_frame(frame):
                state.consecutive_errors += 1
                if state.consecutive_errors > MAX_CONSECUTIVE_ERRORS:
                    camera = state.get_camera()
                    if camera is None:
                        time.sleep(0.5)
                if error_cache is not None:
                    frame = error_cache.copy()
                    cv2.putText(frame, "Stream interrupted...", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                else:
                    continue
            else:
                state.consecutive_errors    = 0
                state.last_successful_frame = time.time()
                error_cache = frame

            frame = cv2.flip(frame, 1)

            # Atomic pointer write — no serialization
            state.raw_slot.write(frame)

            # Push to detection — put_nowait never blocks stream thread
            # frame.copy() is needed so the detect thread gets its own buffer.
            # We only copy on the frame that will actually be processed (every N).
            # On all other frames no copy happens — saves ~921 KB malloc per frame.
            state.frame_count += 1
            if state.frame_count % DETECTION_EVERY_N == 0:
                try:
                    # copy() here is unavoidable (detect thread runs async)
                    # but only happens 1/N of frames — e.g. 1/8 = 12.5% of frames
                    state.detect_queue.put_nowait((frame.copy(), state.current_command))
                except Exception:
                    pass  # Detection busy → drop frame → correct behavior

            # Get pre-encoded bytes from encode thread
            # Falls back to inline encode for first few frames only
            encoded = state.encoded_slot.read()
            if encoded is None:
                ret2, buf = cv2.imencode('.jpg', frame, fallback_params)
                if not ret2:
                    continue
                encoded = buf.tobytes()

            yield HEADER + encoded + FOOTER

            # No sleep — camera.grab() is the natural throttle

        except GeneratorExit:
            break
        except Exception as e:
            print(f"Stream error: {e}")
            state.consecutive_errors += 1


# ============================================================================
# EMBEDDINGS HELPER
# ============================================================================

# ── Embedding cache helpers ───────────────────────────────────────────────────

def _load_hash_cache() -> dict:
    """Load {filepath: (mtime, [norm_embs])} from disk."""
    if os.path.exists(EMBEDDING_CACHE_PATH):
        try:
            with open(EMBEDDING_CACHE_PATH, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass
    return {}


def _save_hash_cache(cache: dict):
    os.makedirs("embeddings", exist_ok=True)
    with open(EMBEDDING_CACHE_PATH, "wb") as f:
        pickle.dump(cache, f)


def _embed_single(img_path: str, person: str, model, cache: dict):
    """
    Process one image. Returns (person, [norm_embeddings], from_cache).
    Called from a thread pool — ONNX Runtime releases the GIL during inference.
    """
    try:
        mtime = os.path.getmtime(img_path)
        entry = cache.get(img_path)
        if entry and entry[0] == mtime:
            return person, entry[1], True
    except OSError:
        pass

    img = cv2.imread(img_path)
    if img is None:
        return person, [], False

    results = []
    try:
        faces = model.get(np.ascontiguousarray(img))
    except Exception:
        return person, [], False

    if not faces:
        return person, [], False

    face = max(faces, key=lambda f: float(getattr(f, "det_score", 0)))
    if float(getattr(face, "det_score", 1.0)) < MIN_FACE_DET_SCORE:
        return person, [], False

    raw = face.embedding.astype(np.float32)
    n   = np.linalg.norm(raw)
    if n < 1e-6:
        return person, [], False
    results.append(raw / n)

    if USE_FLIP_AUGMENT:
        try:
            ff = model.get(np.ascontiguousarray(cv2.flip(img, 1)))
            if ff:
                bf = max(ff, key=lambda f: float(getattr(f, "det_score", 0)))
                if float(getattr(bf, "det_score", 1.0)) >= MIN_FACE_DET_SCORE:
                    fe = bf.embedding.astype(np.float32)
                    fn = np.linalg.norm(fe)
                    if fn > 1e-6:
                        results.append(fe / fn)
        except Exception:
            pass

    try:
        cache[img_path] = (os.path.getmtime(img_path), results)
    except OSError:
        pass

    return person, results, False


def generate_embeddings_from_dataset():
    """
    Parallel, cached embedding generator.
    Speed vs original:
      - File-hash cache   : unchanged images skipped entirely
      - ThreadPoolExecutor: imread + ONNX run in parallel (GIL released)
      - Model reuse       : reuses live recognizer if already loaded
      - Smaller det_size  : (160,160) for offline batch work
      - Best-face pick    : highest det_score not arbitrary faces[0]
      - Flip augment      : doubles effective dataset
      - Per-person mean   : 1 stable centroid → faster runtime cosine search
    """
    if not os.path.exists(DATASET_PATH) or not os.listdir(DATASET_PATH):
        return False, "Dataset folder is empty", 0

    model = state.recognizer
    if model is None:
        for providers in (
            ["CUDAExecutionProvider", "CPUExecutionProvider"],
            ["CPUExecutionProvider"],
        ):
            try:
                model = insightface.app.FaceAnalysis(
                    name="buffalo_l",
                    providers=providers,
                    allowed_modules=["detection", "recognition"]
                )
                model.prepare(ctx_id=0 if "CUDA" in providers[0] else -1,
                              det_size=EMBED_DET_SIZE)
                print(f"Embed model: {providers[0].replace('ExecutionProvider','')}")
                break
            except Exception as e:
                print(f"Provider failed: {e}")

    if model is None:
        return False, "Could not load model", 0

    image_paths = []
    for person in os.listdir(DATASET_PATH):
        pp = os.path.join(DATASET_PATH, person)
        if not os.path.isdir(pp):
            continue
        for img_name in os.listdir(pp):
            if img_name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                image_paths.append((os.path.join(pp, img_name), person))

    if not image_paths:
        return False, "No images found in dataset", 0

    print(f"Processing {len(image_paths)} images "
          f"(workers={EMBED_WORKERS}, flip={USE_FLIP_AUGMENT}, "
          f"aggregate={AGGREGATE_PER_PERSON})...")

    cache       = _load_hash_cache()
    cache_hits  = 0
    person_embs: dict = {}
    processed = failed = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=EMBED_WORKERS) as pool:
        futures = {
            pool.submit(_embed_single, img_path, person, model, cache): (img_path, person)
            for img_path, person in image_paths
        }
        for future in as_completed(futures):
            person, embs, from_cache = future.result()
            if embs:
                person_embs.setdefault(person, []).extend(embs)
                processed += 1
                if from_cache:
                    cache_hits += 1
            else:
                failed += 1

    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s  "
          f"(cache_hits={cache_hits}/{len(image_paths)}, failed={failed})")

    if not person_embs:
        return False, "No valid faces found", 0

    _save_hash_cache(cache)

    embeddings, names = [], []
    if AGGREGATE_PER_PERSON:
        for person, emb_list in person_embs.items():
            stack = np.stack(emb_list, axis=0)
            mean  = stack.mean(axis=0)
            nrm   = np.linalg.norm(mean)
            if nrm > 1e-6:
                embeddings.append(mean / nrm)
                names.append(person)
        print(f"Aggregated {processed} raw embs → {len(names)} person centroid(s)")
    else:
        for person, emb_list in person_embs.items():
            for emb in emb_list:
                embeddings.append(emb)
                names.append(person)

    os.makedirs("embeddings", exist_ok=True)
    with open(EMBEDDINGS_PATH, "wb") as f:
        pickle.dump({"embeddings": embeddings, "names": names,
                     "aggregated": AGGREGATE_PER_PERSON}, f)
    state.load_embeddings()

    n_persons = len(set(names))
    return (True,
            f"Generated {len(embeddings)} embedding(s) for {n_persons} person(s) "
            f"({failed} images skipped) in {elapsed:.1f}s",
            processed)


# ============================================================================
# ROUTES
# ============================================================================

@app.route('/')
def index():
    return render_template('index4.html')

@app.route('/capture')
def capture_page():
    return render_template('capture.html')

@app.route('/manage')
def manage_page():
    persons = []
    if os.path.exists(DATASET_PATH):
        for person in os.listdir(DATASET_PATH):
            pp = os.path.join(DATASET_PATH, person)
            if os.path.isdir(pp):
                count = len([f for f in os.listdir(pp) if f.endswith('.jpg')])
                persons.append({'name': person, 'count': count})
    return render_template('manage.html', persons=persons)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/set_command', methods=['POST'])
def set_command():
    data   = request.json
    result = state.command_parser.parse(data.get('command', ''))
    if result['valid']:
        state.current_command = result
        return jsonify({'success': True,
                        'message': state.command_parser.format_feedback(result),
                        'command': result})
    state.current_command = None
    return jsonify({'success': False,
                    'message': f"Invalid: {result['error']}", 'command': result})

@app.route('/api/clear_command', methods=['POST'])
def clear_command():
    state.current_command       = None
    state.pending_auto_snapshot = None   # drain any crop already queued
    return jsonify({'success': True, 'message': 'Command cleared'})

@app.route('/api/capture_frame', methods=['POST'])
def capture_frame():
    frame, _ = state.raw_slot.read()
    if frame is not None:
        ret, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if ret:
            return jsonify({'success': True,
                            'image': base64.b64encode(buf).decode('utf-8')})
    return jsonify({'success': False, 'message': 'No frame available'})

@app.route('/api/generate_embeddings', methods=['POST'])
def api_generate_embeddings():
    success, message, count = generate_embeddings_from_dataset()
    return jsonify({'success': success, 'message': message, 'count': count})

@app.route('/api/capture_snapshot', methods=['POST'])
def capture_snapshot():
    frame, _ = state.annotated_slot.read()
    if frame is not None:
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"snapshot_{ts}.jpg"
        filepath = os.path.join(SNAPSHOTS_PATH, filename)
        cv2.imwrite(filepath, frame)
        return jsonify({'success': True, 'message': 'Snapshot saved',
                        'filename': filename, 'path': filepath})
    return jsonify({'success': False, 'message': 'No frame available'})

@app.route('/api/get_snapshot/<filename>')
def get_snapshot(filename):
    fp = os.path.join(SNAPSHOTS_PATH, filename)
    if os.path.exists(fp):
        return send_file(fp, mimetype='image/jpeg')
    return jsonify({'error': 'Not found'}), 404

@app.route('/api/detection_status')
def detection_status():
    return jsonify({
        'detection_info':     state.detection_results,
        'current_command':    state.current_command,
        'consecutive_errors': state.consecutive_errors
    })

@app.route('/api/system_status')
def system_status():
    emb_exist  = os.path.exists(EMBEDDINGS_PATH)
    emb_count  = len(state.known_names) if state.known_names else 0
    ds_persons = ([d for d in os.listdir(DATASET_PATH)
                   if os.path.isdir(os.path.join(DATASET_PATH, d))]
                  if os.path.exists(DATASET_PATH) else [])
    return jsonify({
        'embeddings_loaded':  emb_exist,
        'embeddings_count':   emb_count,
        'dataset_persons':    len(ds_persons),
        'persons':            ds_persons,
        'camera_active':      state.camera is not None,
        'current_command':    state.current_command,
        'consecutive_errors': state.consecutive_errors,
        'camera_url':         CAMERA_INDEXES[state.camera_url_index]
    })

@app.route('/api/delete_person/<person_name>', methods=['DELETE'])
def delete_person(person_name):
    pp = os.path.join(DATASET_PATH, person_name)
    if os.path.exists(pp):
        shutil.rmtree(pp)
        return jsonify({'success': True, 'message': f'{person_name} deleted'})
    return jsonify({'success': False, 'message': 'Person not found'})

@app.route('/api/upload_dataset', methods=['POST'])
def upload_dataset():
    try:
        person_name = request.form.get('person_name')
        if not person_name:
            return jsonify({'success': False, 'message': 'Name required'})
        person_dir = os.path.join(DATASET_PATH, person_name)
        os.makedirs(person_dir, exist_ok=True)
        images = request.files.getlist('images')
        if not images:
            return jsonify({'success': False, 'message': 'No images uploaded'})
        saved = 0
        for idx, img in enumerate(images):
            if img:
                img.save(os.path.join(person_dir, f"{idx:04d}.jpg"))
                saved += 1
        return jsonify({'success': True, 'message': f'Saved {saved} images', 'count': saved})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/auto_snapshot', methods=['POST'])
def toggle_auto_snapshot():
    data = request.json or {}
    if 'enabled' in data:
        state.auto_snapshot_enabled = bool(data['enabled'])
        if not state.auto_snapshot_enabled:
            # Disabling resets any pending lock so it's ready next time
            state.snapshot_locked      = False
            state.last_snapshot_person = None
    return jsonify({'success': True,
                    'auto_snapshot_enabled': state.auto_snapshot_enabled})

@app.route('/api/delete_embeddings', methods=['DELETE'])
def delete_embeddings():
    try:
        if os.path.exists(EMBEDDINGS_PATH):
            os.remove(EMBEDDINGS_PATH)
            state.known_embeddings      = None
            state.known_names           = None
            state.normalized_embeddings = None
            return jsonify({'success': True, 'message': 'Embeddings deleted'})
        return jsonify({'success': False, 'message': 'File not found'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/reconnect_camera', methods=['POST'])
def reconnect_camera():
    state.release_camera()
    state.consecutive_errors = 0
    cam = state.get_camera()
    if cam and cam.isOpened():
        return jsonify({'success': True, 'message': 'Camera reconnected'})
    return jsonify({'success': False, 'message': 'Failed to reconnect'})


@app.route('/api/pending_snapshot')
def get_pending_snapshot():
    """Return the latest auto-crop waiting for user verification (then clear it)."""
    snap = state.pending_auto_snapshot
    if snap:
        state.pending_auto_snapshot = None   # consume
        return jsonify({'success': True, 'snapshot': snap})
    return jsonify({'success': False, 'snapshot': None})


# ═══ NEW: Verification & Sketch Routes ═══════════════════════════════════════

@app.route('/api/verify_snapshot', methods=['POST'])
def verify_snapshot():
    """
    Save the verified snapshot with proper naming and generate all 8 sketch
    variants (background removed, full-body isolated).

    Request body:
      {
        "temp_filename":  "auto_Alice_20260219_123456.jpg",
        "person_name":    "Alice",
        "position_desc":  "Person to the right of User1",
        "create_sketch":  true,          // optional, default true
        "company":        "AABBCC"       // optional
      }

    Response includes:
      sketch_filename  — best variant filename (shown first in UI)
      sketch_variants  — list of all 8 variants, sorted best-first:
          [{ filename, variant_num, variant_label, score, is_best }, ...]
    """
    try:
        data          = request.json or {}
        temp_filename = data.get('temp_filename')
        person_name   = data.get('person_name', 'Unknown')
        position_desc = data.get('position_desc', '')
        create_sketch = data.get('create_sketch', True)
        company       = data.get('company', 'AABBCC')

        if not temp_filename:
            return jsonify({'success': False, 'message': 'No filename provided'})

        temp_path = os.path.join(SNAPSHOTS_PATH, temp_filename)
        if not os.path.exists(temp_path):
            return jsonify({'success': False, 'message': 'Snapshot not found'})

        timestamp         = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name         = person_name.replace(" ", "_")
        verified_filename = f"verified_{safe_name}_{timestamp}.jpg"
        verified_path     = os.path.join(SNAPSHOTS_PATH, verified_filename)
        shutil.copy2(temp_path, verified_path)

        result = {
            'success':           True,
            'message':           f'Verified snapshot saved as {verified_filename}',
            'verified_filename': verified_filename,
            'sketch_filename':   None,
            'sketch_variants':   [],
        }

        if create_sketch:
            from sketch_generator_new import generate_sketch_variations
            base_name = f"{safe_name}_{timestamp}"
            variants  = generate_sketch_variations(
                verified_path, SNAPSHOTS_PATH, person_name, company,
                base_name=base_name,
            )

            if variants:
                # Best variant is first (sorted by quality score)
                best = variants[0]
                result['sketch_filename'] = best['filename']
                result['sketch_variants'] = [
                    {
                        'filename':      v['filename'],
                        'variant_num':   v['variant_num'],
                        'variant_label': v['variant_label'],
                        'score':         round(v['score'], 1),
                        'is_best':       v['is_best'],
                    }
                    for v in variants
                ]
                result['message'] += (
                    f' | {len(variants)} sketch variants created'
                    f' (best: v{best["variant_num"]} {best["variant_label"]})'
                )
            else:
                result['message'] += ' | Sketch generation failed'

        return jsonify(result)

    except Exception as e:
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})


@app.route('/api/send_sketch_to_laser', methods=['POST'])
def send_sketch_to_laser():
    """
    Mark the operator-chosen sketch as the final laser file.
    In production, trigger the actual laser engraver job here.

    Request body:
      { "filename": "sketch_Alice_20260319_120000_v3_DeepContrast.jpg" }

    Returns:
      { success, message, filename }
    """
    try:
        data     = request.json or {}
        filename = data.get('filename', '').strip()
        if not filename:
            return jsonify({'success': False, 'message': 'No filename provided'})

        filepath = os.path.join(SNAPSHOTS_PATH, filename)
        if not os.path.exists(filepath):
            return jsonify({'success': False, 'message': 'Sketch file not found'})

        # ── Placeholder: replace this block with your laser SDK call ──────
        print(f"[LASER] Sending to engraver: {filename}")
        # e.g.  laser_sdk.engrave(filepath)

        return jsonify({
            'success':  True,
            'message':  f'Sketch "{filename}" sent to laser engraver',
            'filename': filename,
        })

    except Exception as e:
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})


@app.route('/api/discard_snapshot', methods=['POST'])
def discard_snapshot():
    """Delete a snapshot that was rejected during verification."""
    try:
        data = request.json or {}
        filename = data.get('filename')
        if not filename:
            return jsonify({'success': False, 'message': 'No filename provided'})
        
        filepath = os.path.join(SNAPSHOTS_PATH, filename)
        if os.path.exists(filepath):
            os.remove(filepath)
            return jsonify({'success': True, 'message': 'Snapshot discarded'})
        return jsonify({'success': False, 'message': 'File not found'})
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})


@app.route('/api/reset_snapshot', methods=['POST'])
def reset_snapshot():
    """
    Unlock the snapshot system for a fresh shot.
    Called when operator clicks Retake at any step — wrong person,
    blurry snapshot, or bad sketch — AND after a successful finalise so that
    running the same command again in the same session still fires correctly
    (fix 5: same person must be re-detected after any reset).
    """
    state.snapshot_locked       = False
    state.last_snapshot_person  = None   # always clear so same person triggers again
    state.pending_auto_snapshot = None
    return jsonify({'success': True, 'message': 'Ready for next detection'})


@app.route('/api/clear_command_after_finalise', methods=['POST'])
def clear_command_after_finalise():
    """
    Called after sketch is accepted and sent to laser.
    Clears the current command so the same detection does not auto-fire again
    until operator sets a new command (fix 4).
    Also resets snapshot state so a new run with same person works (fix 5).
    """
    state.current_command       = None
    state.snapshot_locked       = False
    state.last_snapshot_person  = None
    state.pending_auto_snapshot = None
    return jsonify({'success': True, 'message': 'Command cleared after finalise'})


@app.route('/api/speak', methods=['POST'])
def speak_route():
    """
    Trigger a system voice line from the frontend.
    Body: { "text": "Some line to speak" }
    Used at each modal step so the system speaks its confirmation dialogue.
    """
    text = (request.json or {}).get('text', '').strip()
    if not text:
        return jsonify({'success': False, 'message': 'No text provided'})
    state.speak(text)
    return jsonify({'success': True})


@app.route('/api/speaking_state')
def speaking_state():
    """
    Returns whether TTS is currently active (including the post-speech tail).
    Frontend polls this before opening either mic path.
    Response: { "speaking": true/false }
    """
    return jsonify({'speaking': state.is_speaking})


@app.route('/api/voice_command', methods=['POST'])
def voice_command():
    """
    NLP voice command parser — accepts raw spoken text, extracts the command,
    and returns structured result + a guest_hint (name mentioned by operator).

    Body: { "text": "detect the person right to User1" }
    Returns: { success, command, message, guest_hint }
    """
    text = (request.json or {}).get('text', '').strip()
    if not text:
        return jsonify({'success': False, 'message': 'No text provided'})

    # Try parsing the full text as a command first
    result = state.command_parser.parse(text)

    # Extract a guest name hint — any capitalised word not in command keywords
    import re as _re
    _stop = {'detect','find','show','capture','identify','scan','person','people',
             'the','a','an','of','to','on','at','is','are','who','left','right',
             'first','second','third','fourth','standing','sitting','next','side'}
    words  = _re.findall(r"[A-Z][a-z]+|[A-Z]{2,}", text)
    hints  = [w for w in words if w.lower() not in _stop]
    guest_hint = hints[0] if hints else None

    if result['valid']:
        state.current_command = result
        return jsonify({
            'success':    True,
            'message':    state.command_parser.format_feedback(result),
            'command':    result,
            'guest_hint': guest_hint
        })

    return jsonify({
        'success':    False,
        'message':    result.get('error', 'Could not parse command'),
        'command':    result,
        'guest_hint': guest_hint
    })


# ============================================================================
# RUN
# ============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("MAXIMUM FPS FACE RECOGNITION — 3-THREAD PIPELINE + TRACKER")
    print("=" * 70)
    print(f"Threads:         Stream | Detection | Encode (fully decoupled)")
    print(f"Camera:          {CAMERA_INDEXES[0]} (+ {len(CAMERA_INDEXES)-1} fallbacks)")
    print(f"Model:           buffalo_l  (full accuracy, optimized pipeline)")
    print(f"Detection size:  {DETECTION_SIZE} (live) | {SNAPSHOT_DET_SIZE} (snapshot)")
    print(f"Frame scale:     {DETECT_FRAME_SCALE}x  ({int(640*DETECT_FRAME_SCALE)}x{int(480*DETECT_FRAME_SCALE)} detect input)")
    print(f"Detection:       Full inference every {DETECTION_EVERY_N} frames | tracker fills rest")
    print(f"Identity cache:  IoU>{IDENTITY_CACHE_IOU_THRESH} skips cosine search for static faces")
    print(f"Cosine sim:      Fast numpy BLAS (sklearn removed)")
    print(f"Frame validate:  Corner-sampling (200x faster than np.mean)")
    print(f"JPEG encode:     Background thread (zero work on stream hot path)")
    print(f"Stream quality:  {STREAM_JPEG_QUALITY}%")
    print(f"Auto-snapshot:   one-shot script mode")
    print("=" * 70)

    state.initialize_recognizer()
    state.load_embeddings()

    try:
        from waitress import serve
        print("Server: Waitress (production)")
        serve(app, host='0.0.0.0', port=5000, threads=8)
    except ImportError:
        print("Server: Flask dev (pip install waitress for production)")
        app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)
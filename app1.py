from flask import Flask, render_template, Response, request, jsonify, send_file
import base64
import shutil
import cv2
import insightface
import pickle
import numpy as np
from command_parsing_enhanced import CommandParser
from sketch_generator import generate_sketch_with_label
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

RECOGNITION_THRESHOLD  = 0.40   # buffalo_s (MobileNet) produces slightly lower
                                # similarity scores than buffalo_l (ResNet).
                                # 0.40 is the right balance. Raise to 0.45 if
                                # you see false positives (wrong person matched).
# Detection size: (320,320) for CPU-only laptops — fast enough to keep stream smooth
DETECTION_SIZE         = (320, 320)
# Model name — buffalo_s is ~60% faster than buffalo_l on CPU with
# minimal accuracy loss for well-lit close-range scenes.
# Switch back to "buffalo_l" if you need maximum accuracy.
INSIGHTFACE_MODEL      = "buffalo_sc"
# Run detection every N frames — stream always runs at full camera FPS
DETECTION_EVERY_N      = 5          # CPU: run detection every 5th frame, stream stays smooth at full FPS
STREAM_JPEG_QUALITY    = 60          # lower quality = faster JPEG encode on CPU
MAX_CONSECUTIVE_ERRORS  = 10
CAMERA_RECONNECT_BACKOFF_START = 1.0   # seconds — doubles each failed attempt
CAMERA_RECONNECT_BACKOFF_MAX   = 30.0  # cap at 30 seconds between retries
FRAME_VALIDATION_ENABLED = True

# ── Embedding generation config ───────────────────────────────────────────────
EMBEDDING_CACHE_PATH   = "embeddings/file_hash_cache.pkl"  # per-file mtime cache
EMBED_WORKERS          = 2      # keep low on CPU-only laptops to avoid freezing during embedding
EMBED_DET_SIZE         = (320, 320)   # MUST match DETECTION_SIZE so embeddings
                                       # and live inference use identical scale.
                                       # Mismatched scales cause lower cosine similarity
                                       # and recognition failures ("Unknown" for known people).
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


# ══════════════════════════════════════════════════════════════════════════════
# SPLIT STATE — three isolated state classes, each with its own lock.
# Threads only acquire the lock for the state they actually need.
# A slow TTS call never blocks detection writes, and vice versa.
# Lock acquisition order (when two locks are needed): detection → command
# ══════════════════════════════════════════════════════════════════════════════

class DetectionState:
    """
    Everything related to the camera, detected faces, and snapshot management.
    Protected by detection_lock.
    """
    def __init__(self):
        self.detection_lock        = threading.Lock()

        # Camera
        self.camera                = None
        self.camera_lock           = threading.Lock()
        self.consecutive_errors    = 0
        self.last_successful_frame = time.time()
        self.camera_url_index      = 0
        self.frame_count           = 0

        # Lock-free frame slots
        self.raw_slot       = AtomicFrameSlot()
        self.annotated_slot = AtomicFrameSlot()
        self.encoded_slot   = EncodedFrameSlot()

        # Detection queue and results
        self.detect_queue      = Queue(maxsize=1)
        self.detection_results = {}

        # Worker threads
        self.detect_thread   = None
        self.encode_thread   = None
        self.threads_running = False

        # Snapshot state
        self.auto_snapshot_enabled = True
        self.snapshot_locked       = False
        self.last_snapshot_person  = None
        self.pending_auto_snapshot = None

        # Draw buffer
        self._draw_buf = None

        # Pan-hint cache
        self._pan_hint_cache     = None
        self._pan_hint_cache_key = None

        # Embedding model mismatch tracking
        self._embedding_model_mismatch = False
        self._embedding_saved_model    = None

        # Camera health — used by reconnect daemon and frontend
        self.camera_status       = "ok"     # "ok" | "disconnected" | "reconnecting"
        self._reconnect_cancel   = threading.Event()  # set() to stop daemon
        self._reconnect_thread   = None
        # Pre-built error frame (640x480 black with red text)
        self._error_frame        = None


class CommandState:
    """
    Current active command and the parser.
    Protected by command_lock — reads are very fast, contention is negligible.
    """
    def __init__(self):
        self.command_lock   = threading.Lock()
        self.command_parser = CommandParser()
        self.current_command = None


class TTSState:
    """
    Text-to-speech flag and lock.
    Protected by tts_lock.
    """
    def __init__(self):
        self.tts_lock   = threading.Lock()
        self.is_speaking = False


# ── System state ──────────────────────────────────────────────────────────────
class SystemState:
    """
    Composes the three isolated state classes and all business logic methods.
    Route handlers reference state.detection, state.command, state.tts directly
    for fine-grained locking, or use the convenience properties below.
    """

    def __init__(self):
        self.detection = DetectionState()
        self.command   = CommandState()
        self.tts       = TTSState()

        # ── Recognizer lives at the top level (shared read-only after init) ─
        self.recognizer            = None
        self.known_names           = None
        self.normalized_embeddings = None

    # ── Convenience properties (unchanged API for all existing code) ─────────
    # These allow existing route handlers to continue using state.x without
    # needing to know which sub-state they belong to.

    @property
    def camera(self):              return self.detection.camera
    @camera.setter
    def camera(self, v):           self.detection.camera = v

    @property
    def camera_lock(self):         return self.detection.camera_lock

    @property
    def consecutive_errors(self):  return self.detection.consecutive_errors
    @consecutive_errors.setter
    def consecutive_errors(self, v): self.detection.consecutive_errors = v

    @property
    def last_successful_frame(self): return self.detection.last_successful_frame
    @last_successful_frame.setter
    def last_successful_frame(self, v): self.detection.last_successful_frame = v

    @property
    def camera_url_index(self):    return self.detection.camera_url_index
    @camera_url_index.setter
    def camera_url_index(self, v): self.detection.camera_url_index = v

    @property
    def frame_count(self):         return self.detection.frame_count
    @frame_count.setter
    def frame_count(self, v):      self.detection.frame_count = v

    @property
    def raw_slot(self):            return self.detection.raw_slot
    @property
    def annotated_slot(self):      return self.detection.annotated_slot
    @property
    def encoded_slot(self):        return self.detection.encoded_slot
    @property
    def detect_queue(self):        return self.detection.detect_queue

    @property
    def detection_results(self):   return self.detection.detection_results
    @detection_results.setter
    def detection_results(self, v): self.detection.detection_results = v

    @property
    def threads_running(self):     return self.detection.threads_running
    @threads_running.setter
    def threads_running(self, v):  self.detection.threads_running = v

    @property
    def detect_thread(self):       return self.detection.detect_thread
    @detect_thread.setter
    def detect_thread(self, v):    self.detection.detect_thread = v

    @property
    def encode_thread(self):       return self.detection.encode_thread
    @encode_thread.setter
    def encode_thread(self, v):    self.detection.encode_thread = v

    @property
    def auto_snapshot_enabled(self): return self.detection.auto_snapshot_enabled
    @auto_snapshot_enabled.setter
    def auto_snapshot_enabled(self, v): self.detection.auto_snapshot_enabled = v

    @property
    def snapshot_locked(self):     return self.detection.snapshot_locked
    @snapshot_locked.setter
    def snapshot_locked(self, v):  self.detection.snapshot_locked = v

    @property
    def last_snapshot_person(self): return self.detection.last_snapshot_person
    @last_snapshot_person.setter
    def last_snapshot_person(self, v): self.detection.last_snapshot_person = v

    @property
    def pending_auto_snapshot(self): return self.detection.pending_auto_snapshot
    @pending_auto_snapshot.setter
    def pending_auto_snapshot(self, v): self.detection.pending_auto_snapshot = v

    @property
    def _draw_buf(self):           return self.detection._draw_buf
    @_draw_buf.setter
    def _draw_buf(self, v):        self.detection._draw_buf = v

    @property
    def _pan_hint_cache(self):     return self.detection._pan_hint_cache
    @_pan_hint_cache.setter
    def _pan_hint_cache(self, v):  self.detection._pan_hint_cache = v

    @property
    def _pan_hint_cache_key(self): return self.detection._pan_hint_cache_key
    @_pan_hint_cache_key.setter
    def _pan_hint_cache_key(self, v): self.detection._pan_hint_cache_key = v

    @property
    def _embedding_model_mismatch(self): return self.detection._embedding_model_mismatch
    @_embedding_model_mismatch.setter
    def _embedding_model_mismatch(self, v): self.detection._embedding_model_mismatch = v

    @property
    def _embedding_saved_model(self): return self.detection._embedding_saved_model
    @_embedding_saved_model.setter
    def _embedding_saved_model(self, v): self.detection._embedding_saved_model = v

    @property
    def camera_status(self):       return self.detection.camera_status
    @camera_status.setter
    def camera_status(self, v):    self.detection.camera_status = v

    @property
    def command_parser(self):      return self.command.command_parser

    @property
    def current_command(self):     return self.command.current_command
    @current_command.setter
    def current_command(self, v):  self.command.current_command = v

    @property
    def is_speaking(self):         return self.tts.is_speaking
    @is_speaking.setter
    def is_speaking(self, v):      self.tts.is_speaking = v

    @property
    def _tts_lock(self):           return self.tts.tts_lock

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
                    name=INSIGHTFACE_MODEL,
                    providers=providers,
                    allowed_modules=["detection", "recognition"]
                )
                ctx = 0 if "CUDA" in providers[0] else -1
                self.recognizer.prepare(ctx_id=ctx, det_size=DETECTION_SIZE)
                label = "GPU (RTX)" if ctx == 0 else "CPU"
                print(f"✓ InsightFace loaded ({label}) model={INSIGHTFACE_MODEL} det_size={DETECTION_SIZE}")
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
            saved_model = data.get("model", "unknown")
            if saved_model != INSIGHTFACE_MODEL:
                # Model mismatch — embeddings are incompatible with the running model.
                # Do NOT use them: cosine similarity will be near-zero for all faces
                # causing every person to show as "Unknown".
                # Clear embeddings so the frontend warning triggers correctly.
                self.normalized_embeddings = None
                self.known_names           = None
                print(f"⚠ WARNING: Embeddings were generated with '{saved_model}' "
                      f"but current model is '{INSIGHTFACE_MODEL}'.")
                print(f"⚠ Embeddings DISABLED — vector spaces are incompatible.")
                print(f"⚠ Click 'Generate Embeddings' in the app to fix this.")
                # Store mismatch info so /api/system_status can expose it
                self._embedding_model_mismatch = True
                self._embedding_saved_model    = saved_model
                return False
            self._embedding_model_mismatch = False
            self._embedding_saved_model    = saved_model
            print(f"✓ Loaded {len(self.known_names)} faces "
                  f"(model={saved_model})")
            return True
        except Exception as e:
            print(f"❌ Embeddings error: {e}")
            return False

    # ── Text-to-speech ───────────────────────────────────────────────────────

    def speak(self, text: str):
        """
        Speak text aloud via pyttsx3 — non-blocking, never delays detection.
        Uses tts_lock (TTSState) — completely isolated from detection_lock.
        """
        MIC_MUTE_TAIL_MS = 500
        print(f"[SYSTEM] {text}")
        if not _TTS_AVAILABLE:
            return
        def _run():
            with self.tts.tts_lock:
                self.tts.is_speaking = True
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
                    self.tts.is_speaking = False
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
        with self.detection.camera_lock:
            if self.detection.camera is not None and self.detection.camera.isOpened():
                if time.time() - self.detection.last_successful_frame < 5.0:
                    return self.detection.camera
                print("⚠ Camera timeout, reconnecting...")
                self.detection.camera.release()
                self.detection.camera = None

            if self.detection.camera is not None:
                self.detection.camera.release()
                self.detection.camera = None

            for _ in range(len(CAMERA_INDEXES)):
                idx = CAMERA_INDEXES[self.detection.camera_url_index]
                print(f"Trying camera {idx}...")
                self.detection.camera = self._open_camera(idx)
                if self.detection.camera.isOpened():
                    print(f"✓ Camera {idx} connected")
                    self.detection.consecutive_errors    = 0
                    self.detection.last_successful_frame = time.time()
                    return self.detection.camera
                self.detection.camera_url_index = (self.detection.camera_url_index + 1) % len(CAMERA_INDEXES)
                print(f"❌ Camera {idx} failed")

            print("❌ All cameras failed")
            return None

    def release_camera(self):
        with self.detection.camera_lock:
            if self.detection.camera is not None:
                self.detection.camera.release()
                self.detection.camera = None

    # ── Frame validation — fast corner sampling ───────────────────────────────

    @staticmethod
    def validate_frame(frame):
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
        if self.detection.threads_running:
            return
        self.detection.threads_running = True
        self.detection.detect_thread = threading.Thread(
            target=self._detect_worker, daemon=True, name="DetectThread"
        )
        self.detection.encode_thread = threading.Thread(
            target=self._encode_worker, daemon=True, name="EncodeThread"
        )
        self.detection.detect_thread.start()
        self.detection.encode_thread.start()
        print("✓ Detection + Encode threads started")

    # ── Detection worker ──────────────────────────────────────────────────────

    def _detect_worker(self):
        while self.detection.threads_running:
            try:
                frame, command = self.detection.detect_queue.get(timeout=0.05)
            except Empty:
                continue
            try:
                annotated, info = self._run_detection(frame, command)
                self.detection.annotated_slot.write(annotated, info)
                self.detection.detection_results = info
            except Exception as e:
                print(f"Detection error: {e}")
            self.detection.detect_queue.task_done()

    # ── Encode worker ─────────────────────────────────────────────────────────

    def _encode_worker(self):
        params   = [cv2.IMWRITE_JPEG_QUALITY, STREAM_JPEG_QUALITY]
        prev_id  = None
        while self.detection.threads_running:
            annotated, _ = self.detection.annotated_slot.read()
            cur_id = id(annotated) if annotated is not None else None
            if cur_id == prev_id or annotated is None:
                time.sleep(0.002)
                continue
            prev_id = cur_id
            ret, buf = cv2.imencode('.jpg', annotated, params)
            if ret:
                self.detection.encoded_slot.write(buf.tobytes())

    def _build_error_frame(self, message="Camera Disconnected — Reconnecting..."):
        """Build a 640x480 black JPEG error frame shown when camera is unavailable."""
        err = np.zeros((480, 640, 3), dtype=np.uint8)
        # Red banner at top
        cv2.rectangle(err, (0, 0), (640, 70), (30, 0, 0), -1)
        cv2.putText(err, "CAMERA ERROR", (20, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 220), 3)
        # Main message
        cv2.putText(err, message, (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 180), 1)
        cv2.putText(err, "System is retrying automatically...", (20, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 120, 120), 1)
        _, buf = cv2.imencode('.jpg', err, [cv2.IMWRITE_JPEG_QUALITY, 70])
        return buf.tobytes()

    def _start_reconnect_daemon(self):
        """
        Starts a background daemon thread that retries camera connection
        with exponential backoff (1s → 2s → 4s → ... → 30s cap).
        Safe to call multiple times — only one daemon runs at a time.
        """
        with self.detection.detection_lock:
            if (self.detection._reconnect_thread is not None
                    and self.detection._reconnect_thread.is_alive()):
                return  # already running

        self.detection._reconnect_cancel.clear()

        def _daemon():
            backoff = CAMERA_RECONNECT_BACKOFF_START
            self.detection.camera_status = "reconnecting"
            # Pre-build error frame once
            if self.detection._error_frame is None:
                self.detection._error_frame = self._build_error_frame()
            print("🔄 Camera reconnect daemon started")

            while not self.detection._reconnect_cancel.is_set():
                # Try to open camera
                cam = self.get_camera()
                if cam is not None and cam.isOpened():
                    # Warmup: discard first 5 frames to avoid stale buffer
                    for _ in range(5):
                        cam.grab()
                    self.detection.camera_status    = "ok"
                    self.detection.consecutive_errors = 0
                    self.detection._error_frame     = None  # clear error frame
                    print("✓ Camera reconnected by daemon")
                    return

                print(f"🔄 Reconnect failed — retrying in {backoff:.0f}s")
                # Sleep in short intervals so cancel_event is responsive
                elapsed = 0.0
                while elapsed < backoff and not self.detection._reconnect_cancel.is_set():
                    time.sleep(0.5)
                    elapsed += 0.5
                backoff = min(backoff * 2, CAMERA_RECONNECT_BACKOFF_MAX)

            self.detection.camera_status = "disconnected"
            print("🔄 Camera reconnect daemon cancelled")

        t = threading.Thread(target=_daemon, daemon=True, name="CameraReconnectDaemon")
        self.detection._reconnect_thread = t
        t.start()

    def _run_detection(self, frame, command_result):
        if self.recognizer is None:
            self.initialize_recognizer()
        if self.normalized_embeddings is None:
            self.load_embeddings()

        if self.normalized_embeddings is None or len(self.normalized_embeddings) == 0:
            if self._draw_buf is None or self._draw_buf.shape != frame.shape:
                self._draw_buf = np.empty_like(frame)
            np.copyto(self._draw_buf, frame)
            out = self._draw_buf
            cv2.putText(out, "No dataset/identity registered", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return out, {'total_faces': 0, 'message': 'No dataset'}

        fh, fw = frame.shape[:2]

        # Contiguous array avoids InsightFace's internal copy
        faces = self.recognizer.get(np.ascontiguousarray(frame))

        anchor_face    = None
        detected_faces = []

        if faces:
            emb_batch = np.array([f.embedding for f in faces], dtype=np.float32)
            # Single BLAS dgemm — replaces sklearn cosine_similarity
            sims = fast_cosine_batch(emb_batch, self.normalized_embeddings)

            for idx, face in enumerate(faces):
                best_idx   = int(np.argmax(sims[idx]))
                best_score = float(sims[idx, best_idx])
                name = (self.known_names[best_idx]
                        if best_score > RECOGNITION_THRESHOLD else "Unknown")
                bbox     = face.bbox.astype(int)
                x1, y1, x2, y2 = bbox
                center_x = (x1 + x2) >> 1  # faster than //2

                detected_faces.append({
                    "name": name, "score": best_score,
                    "bbox": bbox, "center_x": center_x
                })
                if command_result and name == command_result.get('reference_person'):
                    # Include name and score so single-mode target_face has all fields
                    # Bug fix: anchor_face was missing "name"/"score" → "Unknown" in snapshot popup
                    # and Detection error: 'name' / 'score' in _draw (both same root cause)
                    anchor_face = {"bbox": bbox, "center_x": center_x,
                                   "name": name, "score": best_score}

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
        # Pass confidence score so the UI can show a confidence badge
        confidence = round(float(target_face.get("score", 0.0)) * 100, 1)
        self._save_crop(frame, target_face["bbox"], name, command_result, confidence)
        # NOTE: Do NOT call self.speak() here — the frontend speaks the
        # confirmation question exactly once when it opens the modal.
        # Calling speak() here caused the question to be said twice.

    def _save_crop(self, frame, bbox, person_name, command_result=None, confidence=0.0):
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
                'filename':      fn,
                'person_name':   person_name,
                'position_desc': pos_desc,
                'timestamp':     datetime.now().isoformat(),
                'confidence':    confidence   # recognition score 0–100 for UI badge
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
    On camera failure: yields error frame every 2s while reconnect daemon retries.
    """
    camera = state.get_camera()
    if camera is None:
        # Trigger reconnect daemon immediately
        state._start_reconnect_daemon()

    HEADER = b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
    FOOTER = b'\r\n'

    # If camera unavailable at startup, stream error frames until daemon succeeds
    if camera is None:
        while True:
            if state.detection.camera_status == "ok":
                camera = state.get_camera()
                if camera is not None:
                    break
            # Build error frame if not already built by daemon
            if state.detection._error_frame is None:
                state.detection._error_frame = state._build_error_frame()
            yield HEADER + state.detection._error_frame + FOOTER
            time.sleep(2.0)

    state.start_threads()


    # Fallback encoder only used for very first frames before encode thread warms up
    fallback_params = [cv2.IMWRITE_JPEG_QUALITY, STREAM_JPEG_QUALITY]
    error_cache = None

    while True:
        try:
            # grab() flushes camera buffer and returns immediately if frame ready
            if not camera.grab():
                state.consecutive_errors += 1
                if state.consecutive_errors > MAX_CONSECUTIVE_ERRORS:
                    print("⚠ Camera errors — starting reconnect daemon...")
                    state._start_reconnect_daemon()
                    # Yield error frame while daemon works
                    if state.detection._error_frame is None:
                        state.detection._error_frame = state._build_error_frame()
                    yield HEADER + state.detection._error_frame + FOOTER
                    time.sleep(2.0)
                    # Check if daemon recovered
                    if state.detection.camera_status == "ok":
                        camera = state.get_camera()
                        if camera:
                            state.consecutive_errors = 0
                continue

            ret, frame = camera.retrieve()

            if not ret or not state.validate_frame(frame):
                state.consecutive_errors += 1
                if state.consecutive_errors > MAX_CONSECUTIVE_ERRORS:
                    state._start_reconnect_daemon()
                    if state.detection._error_frame is None:
                        state.detection._error_frame = state._build_error_frame()
                    yield HEADER + state.detection._error_frame + FOOTER
                    time.sleep(2.0)
                    if state.detection.camera_status == "ok":
                        camera = state.get_camera()
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
            state.frame_count += 1
            if state.frame_count % DETECTION_EVERY_N == 0:
                try:
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
                    name=INSIGHTFACE_MODEL,
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
                     "aggregated": AGGREGATE_PER_PERSON,
                     "model": INSIGHTFACE_MODEL}, f)
    state.load_embeddings()
    # load_embeddings clears _embedding_model_mismatch on success

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
    return render_template('index1.html')

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
        'camera_active':        state.camera is not None,
        'embedding_mismatch':   state._embedding_model_mismatch,
        'embedding_model_needed': INSIGHTFACE_MODEL
                                  if state._embedding_model_mismatch else None,
        'camera_status':      state.detection.camera_status,
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
    """Manual reconnect trigger — cancels existing daemon and starts fresh."""
    state.detection._reconnect_cancel.set()   # stop any existing daemon
    state.release_camera()
    state.consecutive_errors = 0
    state.detection.camera_status = "reconnecting"
    state._start_reconnect_daemon()
    return jsonify({'success': True, 'message': 'Reconnect daemon started'})

@app.route('/api/camera_status')
def camera_status_route():
    """Returns current camera health for frontend badge."""
    return jsonify({
        'status': state.detection.camera_status,   # "ok" | "reconnecting" | "disconnected"
        'consecutive_errors': state.consecutive_errors
    })


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
    Save the verified snapshot with proper naming.
    Request body: {
        "temp_filename": "auto_Alice_20260219_123456.jpg",
        "person_name": "Alice",
        "position_desc": "Person to the right of User1",
        "create_sketch": true/false
    }
    """
    try:
        data = request.json or {}
        temp_filename = data.get('temp_filename')
        person_name   = data.get('person_name', 'Unknown')
        position_desc = data.get('position_desc', '')
        create_sketch = data.get('create_sketch', False)
        
        if not temp_filename:
            return jsonify({'success': False, 'message': 'No filename provided'})
        
        # Build paths
        temp_path = os.path.join(SNAPSHOTS_PATH, temp_filename)
        if not os.path.exists(temp_path):
            return jsonify({'success': False, 'message': 'Snapshot not found'})
        
        # Create verified filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = person_name.replace(" ", "_")
        verified_filename = f"verified_{safe_name}_{timestamp}.jpg"
        verified_path     = os.path.join(SNAPSHOTS_PATH, verified_filename)
        
        # Copy/rename the temp file
        shutil.copy2(temp_path, verified_path)
        
        result = {
            'success': True,
            'message': f'Verified snapshot saved as {verified_filename}',
            'verified_filename': verified_filename,
            'sketch_filename': None
        }
        
        # Generate sketch if requested
        if create_sketch:
            sketch_filename = f"sketch_{safe_name}_{timestamp}.jpg"
            sketch_path     = os.path.join(SNAPSHOTS_PATH, sketch_filename)
            
            from sketch_generator import generate_sketch_for_laser
            sketch_success = generate_sketch_for_laser(
                verified_path,
                sketch_path,
                person_name,
                data.get('company', 'AABBCC')
            )
            
            if sketch_success:
                result['sketch_filename'] = sketch_filename
                result['message'] += f' | Sketch created: {sketch_filename}'
            else:
                result['message'] += ' | Sketch generation failed'
        
        return jsonify(result)
        
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
    print("MAXIMUM FPS FACE RECOGNITION — 3-THREAD PIPELINE")
    print("=" * 70)
    print(f"Threads:         Stream | Detection | Encode (fully decoupled)")
    print(f"Camera:          {CAMERA_INDEXES[0]} (+ {len(CAMERA_INDEXES)-1} fallbacks)")
    print(f"Model:           {INSIGHTFACE_MODEL}")
    print(f"Detection:       Every {DETECTION_EVERY_N} frames @ {DETECTION_SIZE}")
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
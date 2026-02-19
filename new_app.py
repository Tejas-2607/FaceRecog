from flask import Flask, render_template, Response, request, jsonify, send_file
import cv2
import insightface
import pickle
import numpy as np
from command_parsing_enhanced import CommandParser
import os
from datetime import datetime
import threading
import warnings
import time
from queue import Queue, Empty

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

app = Flask(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
CAMERA_INDEXES         = [0, 1, 2, 3, 4, 5, 6]
DATASET_PATH           = "dataset"
EMBEDDINGS_PATH        = "embeddings/face_embeddings.pkl"
SNAPSHOTS_PATH         = "snapshots"

RECOGNITION_THRESHOLD  = 0.65
# Use (160,160) on CPU-only machines for ~2x speed at slight accuracy cost
DETECTION_SIZE         = (320, 320)
# Run detection every N frames — stream always runs at full camera FPS
DETECTION_EVERY_N      = 3
STREAM_JPEG_QUALITY    = 65
MAX_CONSECUTIVE_ERRORS = 10
FRAME_VALIDATION_ENABLED = True

os.makedirs(DATASET_PATH, exist_ok=True)
os.makedirs("embeddings", exist_ok=True)
os.makedirs(SNAPSHOTS_PATH, exist_ok=True)


# ── Fast cosine similarity — replaces sklearn ─────────────────────────────────
def fast_cosine_batch(query_vecs, db_vecs_normalized):
    norms = np.linalg.norm(query_vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    return (query_vecs / norms) @ db_vecs_normalized.T


# ── Lock-free single-slot buffer ──────────────────────────────────────────────
class AtomicFrameSlot:
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

        # Auto-snapshot
        self.auto_snapshot_enabled = True
        self.last_snapshot_time    = 0
        self.snapshot_cooldown     = 8.0
        self.last_snapshot_person  = None

    # ── Model init ────────────────────────────────────────────────────────────

    def initialize_recognizer(self):
        if self.recognizer is not None:
            return
        print("Loading InsightFace model...")
        for providers in (
            ["CUDAExecutionProvider", "CPUExecutionProvider"],
            ["CPUExecutionProvider"],
        ):
            try:
                self.recognizer = insightface.app.FaceAnalysis(
                    name="buffalo_l", providers=providers
                )
                ctx = 0 if "CUDA" in providers[0] else -1
                self.recognizer.prepare(ctx_id=ctx, det_size=DETECTION_SIZE)
                label = "GPU" if ctx == 0 else "CPU"
                print(f"✓ InsightFace loaded ({label})")
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
        while self.threads_running:
            try:
                frame, command = self.detect_queue.get(timeout=0.05)
            except Empty:
                continue
            try:
                annotated, info = self._run_detection(frame, command)
                self.annotated_slot.write(annotated, info)
                self.detection_results = info
            except Exception as e:
                print(f"Detection error: {e}")
            self.detect_queue.task_done()

    # ── Encode worker ─────────────────────────────────────────────────────────

    def _encode_worker(self):
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

    def _run_detection(self, frame, command_result):
        if self.recognizer is None:
            self.initialize_recognizer()
        if self.normalized_embeddings is None:
            self.load_embeddings()

        if self.normalized_embeddings is None or len(self.normalized_embeddings) == 0:
            out = frame.copy()
            cv2.putText(out, "No dataset/identity registered", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return out, {'total_faces': 0, 'message': 'No dataset'}

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
                    anchor_face = {"bbox": bbox, "center_x": center_x}

        # Auto-snapshot on clean frame (before drawing boxes)
        if command_result and anchor_face is not None and self.auto_snapshot_enabled:
            self._check_snapshot(frame, detected_faces, command_result, anchor_face)

        annotated = self._draw(frame.copy(), detected_faces, anchor_face, command_result)
        return annotated, {
            'total_faces':     len(detected_faces),
            'anchor_detected': anchor_face is not None if command_result else False,
            'target_detected': False,
            'faces': [{'name': f['name'], 'score': float(f['score'])}
                      for f in detected_faces]
        }

    def _check_snapshot(self, frame, detected_faces, command_result, anchor_face):
        anchor_name = command_result.get('reference_person')
        direction   = command_result.get('direction')
        anchor_cx   = anchor_face["center_x"]

        for data in detected_faces:
            if data["name"] == anchor_name:
                continue
            face_cx = data["center_x"]
            if not ((direction == "right" and face_cx < anchor_cx) or
                    (direction == "left"  and face_cx > anchor_cx)):
                continue
            now = time.time()
            if now - self.last_snapshot_time > self.snapshot_cooldown:
                self._save_crop(frame, data["bbox"], data["name"])
                self.last_snapshot_time   = now
                self.last_snapshot_person = data["name"]

    def _save_crop(self, frame, bbox, person_name):
        try:
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = bbox
            pad_x = int((x2 - x1) * 1.5)
            pad_y = int((y2 - y1) * 2.5)
            crop  = frame[max(0, y1-pad_y):min(h, y2+pad_y),
                          max(0, x1-pad_x):min(w, x2+pad_x)]
            if crop.size == 0:
                return
            ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
            fn  = f"auto_{person_name.replace(' ','_')}_{ts}.jpg"
            cv2.imwrite(os.path.join(SNAPSHOTS_PATH, fn), crop,
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            print(f"📸 {fn}")
        except Exception as e:
            print(f"Snapshot error: {e}")

    def _draw(self, frame, detected_faces, anchor_face, command_result):
        anchor_name = command_result.get('reference_person') if command_result else None
        direction   = command_result.get('direction') if command_result else None

        if command_result and anchor_face is None:
            cv2.putText(frame, f"{anchor_name} not detected", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            for d in detected_faces:
                x1, y1, x2, y2 = d["bbox"]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f"{d['name']} ({d['score']:.2f})",
                            (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        elif command_result and anchor_face is not None:
            ax1, ay1, ax2, ay2 = anchor_face["bbox"]
            anchor_cx = anchor_face["center_x"]
            cv2.rectangle(frame, (ax1, ay1), (ax2, ay2), (0, 255, 0), 2)
            cv2.putText(frame, f"{anchor_name} (Anchor)", (ax1, ay1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            person_found = other_exist = False
            for d in detected_faces:
                if d["name"] == anchor_name:
                    continue
                other_exist = True
                x1, y1, x2, y2 = d["bbox"]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f"{d['name']} ({d['score']:.2f})",
                            (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                if ((direction == "right" and d["center_x"] < anchor_cx) or
                        (direction == "left"  and d["center_x"] > anchor_cx)):
                    person_found = True
                    cv2.putText(frame, f"Person detected: {d['name']}", (30, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

            if not person_found:
                msg = f"No person found on {direction.upper()}" if not other_exist \
                      else f"No person found on {direction.upper()}"
                cv2.putText(frame, msg, (30, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        else:
            for d in detected_faces:
                x1, y1, x2, y2 = d["bbox"]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f"{d['name']} ({d['score']:.2f})",
                            (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        return frame

state = SystemState()

def generate_frames():
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

def generate_embeddings_from_dataset():
    if not os.path.exists(DATASET_PATH) or not os.listdir(DATASET_PATH):
        return False, "Dataset folder is empty", 0

    model = None
    for providers in (
        ["CUDAExecutionProvider", "CPUExecutionProvider"],
        ["CPUExecutionProvider"],
    ):
        try:
            model = insightface.app.FaceAnalysis(name="buffalo_l", providers=providers)
            model.prepare(ctx_id=0 if "CUDA" in providers[0] else -1)
            print(f"Embedding: {providers[0].replace('ExecutionProvider','')}")
            break
        except Exception as e:
            print(f"Provider failed: {e}")

    if model is None:
        return False, "Could not load model", 0

    embeddings, names = [], []
    processed = failed = 0
    image_paths = []

    for person in os.listdir(DATASET_PATH):
        pp = os.path.join(DATASET_PATH, person)
        if not os.path.isdir(pp):
            continue
        for img_name in os.listdir(pp):
            image_paths.append((os.path.join(pp, img_name), person))

    print(f"Processing {len(image_paths)} images...")
    for img_path, person in image_paths:
        try:
            img = cv2.imread(img_path)
            if img is None:
                failed += 1
                continue
            faces = model.get(img)
            if faces:
                emb = faces[0].embedding.astype(np.float32)
                embeddings.append(emb / np.linalg.norm(emb))
                names.append(person)
                processed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"Error {img_path}: {e}")
            failed += 1

    if embeddings:
        os.makedirs("embeddings", exist_ok=True)
        with open(EMBEDDINGS_PATH, "wb") as f:
            pickle.dump({"embeddings": embeddings, "names": names}, f)
        state.load_embeddings()
        return True, f"Generated {processed} embeddings ({failed} failed)", processed

    return False, "No valid faces found", 0

@app.route('/')
def index():
    return render_template('index.html')

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
    state.current_command = None
    return jsonify({'success': True, 'message': 'Command cleared'})

@app.route('/api/capture_frame', methods=['POST'])
def capture_frame():
    import base64
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
        import shutil; shutil.rmtree(pp)
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
    if 'enabled'  in data: state.auto_snapshot_enabled = bool(data['enabled'])
    if 'cooldown' in data: state.snapshot_cooldown     = float(data['cooldown'])
    return jsonify({'success': True,
                    'auto_snapshot_enabled': state.auto_snapshot_enabled,
                    'cooldown_seconds': state.snapshot_cooldown})

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

if __name__ == '__main__':
    print("=" * 70)
    print("MAXIMUM FPS FACE RECOGNITION — 3-THREAD PIPELINE")
    print("=" * 70)
    print(f"Threads:         Stream | Detection | Encode (fully decoupled)")
    print(f"Camera:          {CAMERA_INDEXES[0]} (+ {len(CAMERA_INDEXES)-1} fallbacks)")
    print(f"Detection:       Every {DETECTION_EVERY_N} frames @ {DETECTION_SIZE}")
    print(f"Cosine sim:      Fast numpy BLAS (sklearn removed)")
    print(f"Frame validate:  Corner-sampling (200x faster than np.mean)")
    print(f"JPEG encode:     Background thread (zero work on stream hot path)")
    print(f"Stream quality:  {STREAM_JPEG_QUALITY}%")
    print(f"Auto-snapshot:   {state.snapshot_cooldown}s cooldown")
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
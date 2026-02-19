from flask import Flask, render_template, Response, request, jsonify, send_file
import cv2
import insightface
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from command_parsing_enhanced import CommandParser
import os
from datetime import datetime
import threading
import warnings
import time
from queue import Queue, Empty

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# FIX 1: Remove the global RTSP env var — it interferes with local webcam capture
# os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = ...  ← REMOVED

app = Flask(__name__)

CAMERA_INDEXES = [0] # , 1, 2, 3, 4, 5, 6
CURRENT_CAMERA_INDEX = CAMERA_INDEXES[0]
DATASET_PATH = "dataset"
EMBEDDINGS_PATH = "embeddings/face_embeddings.pkl"
SNAPSHOTS_PATH = "snapshots"

RECOGNITION_THRESHOLD = 0.65
MIN_DETECTION_CONFIDENCE = 0.5

# FIX 2: Increase skip frames — detection every 3rd frame instead of every 2nd.
# At 30 FPS camera, detection runs ~10x/sec which is plenty for recognition.
# The stream thread is NEVER blocked waiting for detection.
DETECTION_SKIP_FRAMES = 2          # process every 3rd frame
STREAM_JPEG_QUALITY = 65           # slight quality drop → faster encode
DETECTION_SIZE = (320, 320)
STREAM_FPS_TARGET = 30
FRAME_BUFFER_SIZE = 1
MAX_CONSECUTIVE_ERRORS = 10
FRAME_VALIDATION_ENABLED = True

os.makedirs(DATASET_PATH, exist_ok=True)
os.makedirs("embeddings", exist_ok=True)
os.makedirs(SNAPSHOTS_PATH, exist_ok=True)


class OptimizedSystemState:
    """Manages global state with error recovery"""

    def __init__(self):
        self.camera = None
        self.recognizer = None
        self.known_embeddings = None
        self.known_names = None
        self.command_parser = CommandParser()
        self.current_command = None

        # FIX 3: Split into TWO locks — one for the annotated frame (written by
        # detection thread, read by stream thread) and one for the raw frame
        # (written by stream thread, read by snapshot/capture endpoints).
        # Previously a single lock caused both threads to block each other.
        self.raw_frame_lock = threading.Lock()
        self.annotated_frame_lock = threading.Lock()
        self.latest_raw_frame = None
        self.latest_annotated_frame = None
        self.detection_results = {}

        self.camera_lock = threading.Lock()
        self.consecutive_errors = 0
        self.last_successful_frame = time.time()
        self.camera_url_index = 0

        self.frame_count = 0
        self.last_detection_result = None

        # FIX 4: Use maxsize=2 so the stream thread never blocks on put_nowait
        # when detection is slow — it just drops the oldest queued frame.
        self.detection_queue = Queue(maxsize=2)
        self.detection_thread = None
        self.detection_running = False

        self.normalized_embeddings = None
        self.auto_snapshot_enabled = True
        self.last_snapshot_time = 0
        self.snapshot_cooldown = 8.0
        self.last_snapshot_person = None

    def initialize_recognizer(self):
        """Load the face recognition model — GPU preferred, CPU fallback"""
        if self.recognizer is None:
            print("Loading InsightFace model...")
            try:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                self.recognizer = insightface.app.FaceAnalysis(
                    name="buffalo_l", providers=providers
                )
                self.recognizer.prepare(ctx_id=0, det_size=DETECTION_SIZE)
                print("✓ InsightFace model loaded (GPU)")
            except Exception as e:
                print(f"GPU init failed ({e}), falling back to CPU")
                self.recognizer = insightface.app.FaceAnalysis(
                    name="buffalo_l", providers=["CPUExecutionProvider"]
                )
                self.recognizer.prepare(ctx_id=-1, det_size=DETECTION_SIZE)
                print("✓ InsightFace model loaded (CPU)")

    def load_embeddings(self):
        """Load and pre-normalize face embeddings"""
        if not os.path.exists(EMBEDDINGS_PATH):
            return False
        try:
            with open(EMBEDDINGS_PATH, "rb") as f:
                data = pickle.load(f)
            embeddings = np.array(data["embeddings"])
            self.known_embeddings = embeddings
            self.normalized_embeddings = embeddings / np.linalg.norm(
                embeddings, axis=1, keepdims=True
            )
            self.known_names = data["names"]
            print(f"✓ Loaded {len(self.known_names)} faces from database")
            return True
        except Exception as e:
            print(f"❌ Error loading embeddings: {e}")
            return False

    def _create_camera_with_options(self, index):
        """Create local webcam with low latency settings"""
        # FIX 5: Try CAP_DSHOW first (Windows DirectShow = lowest latency),
        # but fall back to default backend if DSHOW fails to open the index.
        camera = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not camera.isOpened():
            camera.release()
            camera = cv2.VideoCapture(index)  # fallback: default backend

        if camera.isOpened():
            # FIX 6: Set buffer to 1 to prevent frame accumulation lag.
            # MJPG codec gives 3-5x faster capture decode vs raw YUY2.
            camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            camera.set(cv2.CAP_PROP_FPS, 30)
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        return camera

    def get_camera(self):
        with self.camera_lock:
            if self.camera is not None and self.camera.isOpened():
                if time.time() - self.last_successful_frame < 5.0:
                    return self.camera
                else:
                    print("⚠ Camera timeout detected, reconnecting...")
                    self.camera.release()
                    self.camera = None

            if self.camera is not None:
                self.camera.release()
                self.camera = None

            for attempt in range(len(CAMERA_INDEXES)):
                index = CAMERA_INDEXES[self.camera_url_index]
                print(f"Trying camera index: {index}")
                self.camera = self._create_camera_with_options(index)

                if self.camera.isOpened():
                    print(f"✓ Camera {index} connected")
                    self.consecutive_errors = 0
                    self.last_successful_frame = time.time()
                    return self.camera

                self.camera_url_index = (self.camera_url_index + 1) % len(CAMERA_INDEXES)
                print(f"❌ Camera {index} failed, trying next...")

            print("❌ All camera indexes failed")
            return None

    def validate_frame(self, frame):
        if frame is None or frame.size == 0:
            return False
        if FRAME_VALIDATION_ENABLED:
            mean_val = np.mean(frame)
            if mean_val < 5 or mean_val > 250:
                return False
        return True

    def release_camera(self):
        with self.camera_lock:
            if self.camera is not None:
                self.camera.release()
                self.camera = None
                print("Camera released")

    def start_detection_thread(self):
        if self.detection_thread is None or not self.detection_thread.is_alive():
            self.detection_running = True
            self.detection_thread = threading.Thread(
                target=self._detection_worker, daemon=True
            )
            self.detection_thread.start()
            print("✓ Detection thread started")

    def _detection_worker(self):
        """Worker thread for async face detection — runs independently of stream"""
        while self.detection_running:
            try:
                frame, command = self.detection_queue.get(timeout=0.1)
                annotated, info = self._process_detection(frame, command)

                # FIX 7: Use the dedicated annotated_frame_lock (not the shared
                # frame_lock) so the stream thread is never blocked here.
                with self.annotated_frame_lock:
                    self.latest_annotated_frame = annotated
                    self.detection_results = info
                    self.last_detection_result = (annotated, info)

                self.detection_queue.task_done()

            except Empty:
                continue
            except Exception as e:
                print(f"Detection worker error: {e}")

    def _process_detection(self, frame, command_result):
        """Optimized detection processing"""
        if self.recognizer is None:
            self.initialize_recognizer()
        if self.normalized_embeddings is None:
            self.load_embeddings()

        if self.normalized_embeddings is None or len(self.normalized_embeddings) == 0:
            out = frame.copy()
            cv2.putText(out, "No dataset/identity registered", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return out, {'total_faces': 0, 'message': 'No dataset'}

        faces = self.recognizer.get(frame)
        anchor_face = None
        detected_faces = []

        if len(faces) > 0:
            embeddings_batch = np.array([face.embedding for face in faces])
            embeddings_batch = embeddings_batch / np.linalg.norm(
                embeddings_batch, axis=1, keepdims=True
            )
            all_sims = cosine_similarity(embeddings_batch, self.normalized_embeddings)

            for idx, face in enumerate(faces):
                sims = all_sims[idx]
                best_idx = np.argmax(sims)
                best_score = sims[best_idx]
                name = (
                    self.known_names[best_idx]
                    if best_score > RECOGNITION_THRESHOLD
                    else "Unknown"
                )
                bbox = face.bbox.astype(int)
                x1, y1, x2, y2 = bbox
                center_x = (x1 + x2) // 2
                detected_faces.append({
                    "name": name, "score": best_score,
                    "bbox": bbox, "center_x": center_x
                })
                if command_result and name == command_result.get('reference_person'):
                    anchor_face = {"bbox": bbox, "center_x": center_x}

        # Auto-snapshot — uses raw frame (no boxes drawn yet)
        if command_result and anchor_face is not None and self.auto_snapshot_enabled:
            anchor_name = command_result.get('reference_person')
            direction   = command_result.get('direction')
            anchor_cx   = anchor_face["center_x"]

            for data in detected_faces:
                if data["name"] == anchor_name:
                    continue
                face_cx  = data["center_x"]
                is_match = (
                    (direction == "right" and face_cx < anchor_cx) or
                    (direction == "left"  and face_cx > anchor_cx)
                )
                if is_match:
                    now        = time.time()
                    person_key = data["name"]
                    cooldown_ok = (
                        now - self.last_snapshot_time > self.snapshot_cooldown or
                        self.last_snapshot_person != person_key
                    )
                    if cooldown_ok:
                        self._save_clean_crop(frame, data["bbox"], data["name"])
                        self.last_snapshot_time   = now
                        self.last_snapshot_person = person_key

        # Draw on a copy so the raw frame stays clean
        annotated_frame = self._draw_detections(
            frame.copy(), detected_faces, anchor_face, command_result
        )
        detection_info = {
            'total_faces': len(detected_faces),
            'anchor_detected': anchor_face is not None if command_result else False,
            'target_detected': False,
            'faces': [{'name': f['name'], 'score': float(f['score'])} for f in detected_faces]
        }
        return annotated_frame, detection_info

    def _save_clean_crop(self, frame, bbox, person_name):
        """Crop detected person with padding, no boxes, save to snapshots."""
        try:
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = bbox
            pad_x  = int((x2 - x1) * 1.5)
            pad_y  = int((y2 - y1) * 2.5)
            x1_pad = max(0, x1 - pad_x)
            y1_pad = max(0, y1 - pad_y)
            x2_pad = min(w, x2 + pad_x)
            y2_pad = min(h, y2 + pad_y)
            crop   = frame[y1_pad:y2_pad, x1_pad:x2_pad]
            if crop.size == 0:
                return
            timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_name  = person_name.replace(" ", "_")
            filename   = f"auto_{safe_name}_{timestamp}.jpg"
            filepath   = os.path.join(SNAPSHOTS_PATH, filename)
            cv2.imwrite(filepath, crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
            print(f"📸 Auto-snapshot saved: {filename}")
        except Exception as e:
            print(f"Auto-snapshot error: {e}")

    def _draw_detections(self, frame, detected_faces, anchor_face, command_result):
        """Draw bounding boxes and labels"""
        anchor_name = command_result.get('reference_person') if command_result else None
        direction   = command_result.get('direction') if command_result else None

        if command_result and anchor_face is None:
            cv2.putText(frame, f"{anchor_name} not detected", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            for data in detected_faces:
                x1, y1, x2, y2 = data["bbox"]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f"{data['name']} ({data['score']:.2f})",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        elif command_result and anchor_face is not None:
            ax1, ay1, ax2, ay2 = anchor_face["bbox"]
            anchor_center_x = anchor_face["center_x"]
            cv2.rectangle(frame, (ax1, ay1), (ax2, ay2), (0, 255, 0), 2)
            cv2.putText(frame, f"{anchor_name} (Anchor)", (ax1, ay1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            person_found     = False
            other_faces_exist = False

            for data in detected_faces:
                name        = data["name"]
                score       = data["score"]
                x1, y1, x2, y2 = data["bbox"]
                face_center_x  = data["center_x"]
                if name == anchor_name:
                    continue
                other_faces_exist = True
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f"{name} ({score:.2f})", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                if direction == "right" and face_center_x < anchor_center_x:
                    person_found = True
                    cv2.putText(frame, f"Person detected: {name}", (30, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                elif direction == "left" and face_center_x > anchor_center_x:
                    person_found = True
                    cv2.putText(frame, f"Person detected: {name}", (30, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

            if not person_found:
                if not other_faces_exist:
                    cv2.putText(frame, "Move the camera", (30, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                else:
                    msg = "No person found on RIGHT" if direction == "right" else "No person found on LEFT"
                    cv2.putText(frame, msg, (30, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        else:
            for data in detected_faces:
                x1, y1, x2, y2 = data["bbox"]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f"{data['name']} ({data['score']:.2f})",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        return frame


state = OptimizedSystemState()


# ============================================================================
# STREAM — completely decoupled from detection timing
# ============================================================================

def generate_frames():
    """
    FIX 8: The stream loop NEVER waits for detection to finish.
    It reads whatever the latest annotated frame is (cached by detection thread)
    and immediately moves on. Detection runs in parallel at its own pace.
    The sleep is removed entirely — let the camera's natural read speed set FPS.
    """
    camera = state.get_camera()
    if camera is None:
        error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(error_frame, "Camera Error - Check Connection", (50, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        _, buffer = cv2.imencode('.jpg', error_frame,
                                 [cv2.IMWRITE_JPEG_QUALITY, STREAM_JPEG_QUALITY])
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
               + buffer.tobytes() + b'\r\n')
        return

    state.start_detection_thread()
    error_frame_cache = None

    # FIX 9: Pre-allocate the JPEG encode params list once, not every frame
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, STREAM_JPEG_QUALITY]

    while True:
        try:
            # FIX 10: Use grab() + retrieve() split — grab() is non-blocking and
            # flushes the camera buffer, retrieve() decodes only when needed.
            if not camera.grab():
                state.consecutive_errors += 1
                if state.consecutive_errors > MAX_CONSECUTIVE_ERRORS:
                    print("⚠ Too many grab errors, reconnecting...")
                    camera = state.get_camera()
                    if camera is None:
                        time.sleep(1.0)
                continue

            ret, frame = camera.retrieve()

            if not ret or not state.validate_frame(frame):
                state.consecutive_errors += 1
                if state.consecutive_errors > MAX_CONSECUTIVE_ERRORS:
                    print("⚠ Invalid frames, reconnecting...")
                    camera = state.get_camera()
                    if camera is None:
                        time.sleep(1.0)
                if error_frame_cache is not None:
                    frame = error_frame_cache.copy()
                    cv2.putText(frame, "Stream interrupted...", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                else:
                    continue
            else:
                state.consecutive_errors = 0
                state.last_successful_frame = time.time()
                error_frame_cache = frame  # no copy needed — we flip into new mem below

            # Mirror — cv2.flip always returns a new array
            frame = cv2.flip(frame, 1)

            # FIX 11: Store raw frame with raw_frame_lock (never blocks detection)
            with state.raw_frame_lock:
                state.latest_raw_frame = frame

            # FIX 12: Push to detection queue without copying if queue has space.
            # put_nowait never blocks the stream thread.
            state.frame_count += 1
            if (state.frame_count % (DETECTION_SKIP_FRAMES + 1)) == 0:
                try:
                    # Only copy here — detection needs its own buffer
                    state.detection_queue.put_nowait((frame.copy(), state.current_command))
                except Exception:
                    pass  # queue full → detection busy → skip this frame, that's fine

            # FIX 13: Read annotated frame with its own lock — never contends with
            # raw_frame_lock used by API endpoints.
            with state.annotated_frame_lock:
                display_frame = (
                    state.latest_annotated_frame
                    if state.latest_annotated_frame is not None
                    else frame
                )

            ret, buffer = cv2.imencode('.jpg', display_frame, encode_params)
            if not ret:
                continue

            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
                   + buffer.tobytes() + b'\r\n')

            # FIX 14: NO sleep here. The camera.grab() call itself is the natural
            # throttle — it blocks only as long as the camera buffer needs.
            # Adding sleep on top was the single biggest FPS killer.

        except GeneratorExit:
            break
        except Exception as e:
            print(f"Stream error: {e}")
            state.consecutive_errors += 1


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def generate_embeddings_from_dataset():
    """Generate embeddings with error handling"""
    if not os.path.exists(DATASET_PATH) or not os.listdir(DATASET_PATH):
        return False, "Dataset folder is empty", 0

    try:
        model = insightface.app.FaceAnalysis(
            name="buffalo_l",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        model.prepare(ctx_id=0)
        print("Using GPU for embedding generation")
    except Exception as e:
        print(f"GPU unavailable ({e}), using CPU")
        model = insightface.app.FaceAnalysis(
            name="buffalo_l", providers=["CPUExecutionProvider"]
        )
        model.prepare(ctx_id=-1)

    embeddings, names = [], []
    processed = failed = 0
    image_paths = []

    for person in os.listdir(DATASET_PATH):
        person_path = os.path.join(DATASET_PATH, person)
        if not os.path.isdir(person_path):
            continue
        for img_name in os.listdir(person_path):
            image_paths.append((os.path.join(person_path, img_name), person))

    print(f"Processing {len(image_paths)} images...")
    for img_path, person in image_paths:
        try:
            img = cv2.imread(img_path)
            if img is None:
                failed += 1
                continue
            faces = model.get(img)
            if len(faces) > 0:
                emb = faces[0].embedding
                emb = emb / np.linalg.norm(emb)
                embeddings.append(emb)
                names.append(person)
                processed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            failed += 1

    if embeddings:
        data = {"embeddings": embeddings, "names": names}
        os.makedirs("embeddings", exist_ok=True)
        with open(EMBEDDINGS_PATH, "wb") as f:
            pickle.dump(data, f)
        state.load_embeddings()
        return True, f"Generated {processed} embeddings ({failed} failed)", processed

    return False, "No valid faces found", 0


# ============================================================================
# ROUTES
# ============================================================================

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
            person_path = os.path.join(DATASET_PATH, person)
            if os.path.isdir(person_path):
                count = len([f for f in os.listdir(person_path) if f.endswith('.jpg')])
                persons.append({'name': person, 'count': count})
    return render_template('manage.html', persons=persons)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/set_command', methods=['POST'])
def set_command():
    data = request.json
    command_text = data.get('command', '')
    result = state.command_parser.parse(command_text)
    if result['valid']:
        state.current_command = result
        feedback = state.command_parser.format_feedback(result)
        return jsonify({'success': True, 'message': feedback, 'command': result})
    state.current_command = None
    return jsonify({'success': False, 'message': f"Invalid: {result['error']}", 'command': result})

@app.route('/api/clear_command', methods=['POST'])
def clear_command():
    state.current_command = None
    return jsonify({'success': True, 'message': 'Command cleared'})

@app.route('/api/capture_frame', methods=['POST'])
def capture_frame():
    import base64
    # FIX 15: Use raw_frame_lock (not the old shared frame_lock)
    with state.raw_frame_lock:
        if state.latest_raw_frame is not None:
            ret, buffer = cv2.imencode('.jpg', state.latest_raw_frame,
                                       [cv2.IMWRITE_JPEG_QUALITY, 90])
            if ret:
                img_base64 = base64.b64encode(buffer).decode('utf-8')
                return jsonify({'success': True, 'image': img_base64})
    return jsonify({'success': False, 'message': 'No frame available'})

@app.route('/api/generate_embeddings', methods=['POST'])
def api_generate_embeddings():
    success, message, count = generate_embeddings_from_dataset()
    return jsonify({'success': success, 'message': message, 'count': count})

@app.route('/api/capture_snapshot', methods=['POST'])
def capture_snapshot():
    # FIX 16: Use annotated_frame_lock
    with state.annotated_frame_lock:
        if state.latest_annotated_frame is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename  = f"snapshot_{timestamp}.jpg"
            filepath  = os.path.join(SNAPSHOTS_PATH, filename)
            cv2.imwrite(filepath, state.latest_annotated_frame)
            return jsonify({'success': True, 'message': 'Snapshot saved',
                            'filename': filename, 'path': filepath})
    return jsonify({'success': False, 'message': 'No frame available'})

@app.route('/api/get_snapshot/<filename>')
def get_snapshot(filename):
    filepath = os.path.join(SNAPSHOTS_PATH, filename)
    if os.path.exists(filepath):
        return send_file(filepath, mimetype='image/jpeg')
    return jsonify({'error': 'Not found'}), 404

@app.route('/api/detection_status')
def detection_status():
    with state.annotated_frame_lock:
        return jsonify({
            'detection_info': state.detection_results,
            'current_command': state.current_command,
            'consecutive_errors': state.consecutive_errors
        })

@app.route('/api/system_status')
def system_status():
    embeddings_exist = os.path.exists(EMBEDDINGS_PATH)
    embeddings_count = len(state.known_names) if state.known_names else 0
    dataset_persons  = []
    if os.path.exists(DATASET_PATH):
        dataset_persons = [
            d for d in os.listdir(DATASET_PATH)
            if os.path.isdir(os.path.join(DATASET_PATH, d))
        ]
    return jsonify({
        'embeddings_loaded':  embeddings_exist,
        'embeddings_count':   embeddings_count,
        'dataset_persons':    len(dataset_persons),
        'persons':            dataset_persons,
        'camera_active':      state.camera is not None,
        'current_command':    state.current_command,
        'consecutive_errors': state.consecutive_errors,
        'camera_url':         CAMERA_INDEXES[state.camera_url_index]
    })

@app.route('/api/delete_person/<person_name>', methods=['DELETE'])
def delete_person(person_name):
    person_path = os.path.join(DATASET_PATH, person_name)
    if os.path.exists(person_path):
        import shutil
        shutil.rmtree(person_path)
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
        saved_count = 0
        for idx, image in enumerate(images):
            if image:
                filepath = os.path.join(person_dir, f"{idx:04d}.jpg")
                image.save(filepath)
                saved_count += 1
        return jsonify({'success': True, 'message': f'Saved {saved_count} images',
                        'count': saved_count})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/auto_snapshot', methods=['POST'])
def toggle_auto_snapshot():
    data = request.json or {}
    if 'enabled' in data:
        state.auto_snapshot_enabled = bool(data['enabled'])
    if 'cooldown' in data:
        state.snapshot_cooldown = float(data['cooldown'])
    return jsonify({
        'success': True,
        'auto_snapshot_enabled': state.auto_snapshot_enabled,
        'cooldown_seconds': state.snapshot_cooldown
    })

@app.route('/api/delete_embeddings', methods=['DELETE'])
def delete_embeddings():
    try:
        if os.path.exists(EMBEDDINGS_PATH):
            os.remove(EMBEDDINGS_PATH)
            state.known_embeddings      = None
            state.known_names           = None
            state.normalized_embeddings = None
            return jsonify({'success': True, 'message': 'Embeddings file deleted successfully'})
        return jsonify({'success': False, 'message': 'Embeddings file not found'})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error deleting embeddings: {str(e)}'})

@app.route('/api/reconnect_camera', methods=['POST'])
def reconnect_camera():
    state.release_camera()
    state.consecutive_errors = 0
    camera = state.get_camera()
    if camera and camera.isOpened():
        return jsonify({'success': True, 'message': 'Camera reconnected'})
    return jsonify({'success': False, 'message': 'Failed to reconnect'})


# ============================================================================
# RUN
# ============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("ULTRA-LOW LATENCY FACE RECOGNITION WITH ERROR RECOVERY")
    print("=" * 70)
    print(f"Primary Camera:  {CAMERA_INDEXES[0]}")
    print(f"Fallback count:  {len(CAMERA_INDEXES) - 1}")
    print(f"Dataset:         {DATASET_PATH}")
    print(f"Detection:       Every {DETECTION_SKIP_FRAMES + 1} frames @ {DETECTION_SIZE}")
    print(f"Stream:          {STREAM_FPS_TARGET} FPS @ {STREAM_JPEG_QUALITY}% quality")
    print(f"Error Recovery:  Reconnect after {MAX_CONSECUTIVE_ERRORS} errors")
    print(f"Frame Validation:{'Enabled' if FRAME_VALIDATION_ENABLED else 'Disabled'}")
    print("=" * 70)

    state.initialize_recognizer()
    state.load_embeddings()

    try:
        from waitress import serve
        print("Using Waitress production server")
        serve(app, host='0.0.0.0', port=5000, threads=8)
    except ImportError:
        print("Using Flask dev server (install waitress for better performance)")
        app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)
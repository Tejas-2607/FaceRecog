from flask import Flask, render_template, Response, request, jsonify, send_file
import cv2
import insightface
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from command_parsing_enhanced1 import CommandParser
import os
from datetime import datetime
import threading
import warnings
from collections import deque
import time
from queue import Queue, Empty

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

app = Flask(__name__)

# ============================================================================
# OPTIMIZED CONFIGURATION
# ============================================================================

CAMERA_INDEX = "rtsp://admin:Intern@123@192.168.0.60:554/stream2"

# Paths
DATASET_PATH = "dataset"
EMBEDDINGS_PATH = "embeddings/face_embeddings.pkl"
SNAPSHOTS_PATH = "snapshots"

# Optimized Recognition settings
RECOGNITION_THRESHOLD = 0.65
MIN_DETECTION_CONFIDENCE = 0.5

# Performance optimizations
DETECTION_SKIP_FRAMES = 2  # Process every 3rd frame (0=every frame, 1=every 2nd, 2=every 3rd)
STREAM_JPEG_QUALITY = 75  # Lower quality = faster encoding
DETECTION_SIZE = (480, 480)  # Smaller detection size = faster processing
STREAM_FPS_TARGET = 20  # Target FPS for streaming
FRAME_BUFFER_SIZE = 2  # Keep only latest frames

# Create necessary directories
os.makedirs(DATASET_PATH, exist_ok=True)
os.makedirs("embeddings", exist_ok=True)
os.makedirs(SNAPSHOTS_PATH, exist_ok=True)

# ============================================================================
# OPTIMIZED GLOBAL STATE
# ============================================================================

class OptimizedSystemState:
    """Manages global state with optimizations"""
    
    def __init__(self):
        self.camera = None
        self.recognizer = None
        self.known_embeddings = None
        self.known_names = None
        self.command_parser = CommandParser()
        self.current_command = None
        
        # Optimized frame handling
        self.frame_lock = threading.Lock()
        self.latest_raw_frame = None
        self.latest_annotated_frame = None
        self.detection_results = {}
        
        # Camera handling
        self.camera_lock = threading.Lock()
        
        # Frame skipping
        self.frame_count = 0
        self.last_detection_result = None
        
        # Detection queue for async processing
        self.detection_queue = Queue(maxsize=1)
        self.detection_thread = None
        self.detection_running = False
        
        # Cached normalized embeddings (faster similarity computation)
        self.normalized_embeddings = None
        
    def initialize_recognizer(self):
        """Load the face recognition model with GPU support"""
        if self.recognizer is None:
            print("Loading InsightFace model...")
            
            # Try GPU first, fallback to CPU
            try:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                self.recognizer = insightface.app.FaceAnalysis(
                    name="buffalo_l",
                    providers=providers
                )
                self.recognizer.prepare(ctx_id=0, det_size=DETECTION_SIZE)
                print("✓ InsightFace model loaded (GPU)")
            except:
                self.recognizer = insightface.app.FaceAnalysis(
                    name="buffalo_l",
                    providers=["CPUExecutionProvider"]
                )
                self.recognizer.prepare(ctx_id=-1, det_size=DETECTION_SIZE)
                print("✓ InsightFace model loaded (CPU)")
    
    def load_embeddings(self):
        """Load and pre-normalize face embeddings for faster comparison"""
        if not os.path.exists(EMBEDDINGS_PATH):
            return False
        
        try:
            with open(EMBEDDINGS_PATH, "rb") as f:
                data = pickle.load(f)
            
            # Pre-normalize embeddings (only done once)
            embeddings = np.array(data["embeddings"])
            self.known_embeddings = embeddings
            self.normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            self.known_names = data["names"]
            
            print(f"✓ Loaded {len(self.known_names)} faces from database (pre-normalized)")
            return True
        except Exception as e:
            print(f"❌ Error loading embeddings: {e}")
            return False
    
    def get_camera(self):
        """Get or initialize camera with optimized settings"""
        with self.camera_lock:
            if self.camera is not None and self.camera.isOpened():
                return self.camera
            
            if self.camera is not None:
                self.camera.release()
            
            print(f"Opening camera {CAMERA_INDEX}...")
            self.camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_FFMPEG)
            
            if not self.camera.isOpened():
                print(f"❌ Cannot open camera {CAMERA_INDEX}")
                return None
            
            # Optimized camera properties
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer lag
            self.camera.set(cv2.CAP_PROP_FPS, STREAM_FPS_TARGET)
            
            print("✓ Camera opened successfully")
            return self.camera
    
    def release_camera(self):
        """Release camera resource"""
        with self.camera_lock:
            if self.camera is not None:
                self.camera.release()
                self.camera = None
                print("Camera released")
    
    def start_detection_thread(self):
        """Start async detection processing thread"""
        if self.detection_thread is None or not self.detection_thread.is_alive():
            self.detection_running = True
            self.detection_thread = threading.Thread(target=self._detection_worker, daemon=True)
            self.detection_thread.start()
            print("✓ Detection thread started")
    
    def _detection_worker(self):
        """Worker thread for async face detection"""
        while self.detection_running:
            try:
                # Get frame from queue with timeout
                frame, command = self.detection_queue.get(timeout=0.1)
                
                # Process detection
                annotated, info = self._process_detection(frame, command)
                
                # Store result
                with self.frame_lock:
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
            cv2.putText(frame, "No dataset/identity registered", (30, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return frame, {'total_faces': 0, 'message': 'No dataset'}
        
        # Detect faces
        faces = self.recognizer.get(frame)
        
        anchor_face = None
        detected_faces = []
        
        # Optimized recognition with vectorized operations
        if len(faces) > 0:
            # Batch process all embeddings at once
            embeddings_batch = np.array([face.embedding for face in faces])
            embeddings_batch = embeddings_batch / np.linalg.norm(embeddings_batch, axis=1, keepdims=True)
            
            # Compute all similarities at once (vectorized)
            all_sims = cosine_similarity(embeddings_batch, self.normalized_embeddings)
            
            for idx, face in enumerate(faces):
                sims = all_sims[idx]
                best_idx = np.argmax(sims)
                best_score = sims[best_idx]
                
                name = self.known_names[best_idx] if best_score > RECOGNITION_THRESHOLD else "Unknown"
                
                bbox = face.bbox.astype(int)
                x1, y1, x2, y2 = bbox
                center_x = (x1 + x2) // 2
                
                detected_faces.append({
                    "name": name,
                    "score": best_score,
                    "bbox": bbox,
                    "center_x": center_x
                })
                
                if command_result and name == command_result.get('reference_person'):
                    anchor_face = {"bbox": bbox, "center_x": center_x}
        
        # Drawing logic (same as original)
        frame = self._draw_detections(frame, detected_faces, anchor_face, command_result)
        
        detection_info = {
            'total_faces': len(detected_faces),
            'anchor_detected': anchor_face is not None if command_result else False,
            'target_detected': False,
            'faces': [{'name': f['name'], 'score': float(f['score'])} for f in detected_faces]
        }
        
        return frame, detection_info
    
    def _draw_detections(self, frame, detected_faces, anchor_face, command_result):
        """Draw bounding boxes and labels (same logic as original)"""
        anchor_name = command_result.get('reference_person') if command_result else None
        direction = command_result.get('direction') if command_result else None
        
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
            
            person_found = False
            other_faces_exist = False
            
            for data in detected_faces:
                name = data["name"]
                score = data["score"]
                x1, y1, x2, y2 = data["bbox"]
                face_center_x = data["center_x"]
                
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
                    if direction == "right":
                        cv2.putText(frame, "No person found on RIGHT", (30, 80),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    elif direction == "left":
                        cv2.putText(frame, "No person found on LEFT", (30, 80),
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
# OPTIMIZED VIDEO STREAMING
# ============================================================================

def generate_frames():
    """Optimized frame generation with frame skipping"""
    camera = state.get_camera()
    if camera is None:
        error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(error_frame, "Camera Error - Check CAMERA_INDEX", (50, 240),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        ret, buffer = cv2.imencode('.jpg', error_frame, [cv2.IMWRITE_JPEG_QUALITY, STREAM_JPEG_QUALITY])
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        return
    
    # Start detection thread
    state.start_detection_thread()
    
    frame_time = 1.0 / STREAM_FPS_TARGET
    last_time = time.time()
    
    while True:
        try:
            # Read frame
            ret, frame = camera.read()
            if not ret:
                time.sleep(0.05)
                continue
            
            # Mirror frame
            frame = cv2.flip(frame, 1)
            
            # Store raw frame
            with state.frame_lock:
                state.latest_raw_frame = frame.copy()
            
            # Frame skipping logic
            state.frame_count += 1
            should_detect = (state.frame_count % (DETECTION_SKIP_FRAMES + 1)) == 0
            
            if should_detect:
                # Queue frame for async detection (non-blocking)
                try:
                    state.detection_queue.put_nowait((frame.copy(), state.current_command))
                except:
                    pass  # Queue full, skip this frame
            
            # Use last detection result for annotation
            with state.frame_lock:
                if state.latest_annotated_frame is not None:
                    display_frame = state.latest_annotated_frame
                else:
                    display_frame = frame
            
            # Fast JPEG encoding
            ret, buffer = cv2.imencode('.jpg', display_frame,
                                      [cv2.IMWRITE_JPEG_QUALITY, STREAM_JPEG_QUALITY])
            if not ret:
                continue
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            
            # Frame rate limiting
            elapsed = time.time() - last_time
            sleep_time = max(0, frame_time - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
            last_time = time.time()
            
        except GeneratorExit:
            break
        except Exception as e:
            print(f"Stream error: {e}")
            time.sleep(0.1)


# ============================================================================
# OPTIMIZED HELPER FUNCTIONS
# ============================================================================

def generate_embeddings_from_dataset():
    """Optimized embedding generation with batch processing"""
    if not os.path.exists(DATASET_PATH) or not os.listdir(DATASET_PATH):
        return False, "Dataset folder is empty", 0
    
    # Use GPU if available
    try:
        model = insightface.app.FaceAnalysis(
            name="buffalo_l",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        model.prepare(ctx_id=0)
        print("Using GPU for embedding generation")
    except:
        model = insightface.app.FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"]
        )
        model.prepare(ctx_id=-1)
        print("Using CPU for embedding generation")
    
    embeddings = []
    names = []
    processed = 0
    failed = 0
    
    # Collect all images first
    image_paths = []
    for person in os.listdir(DATASET_PATH):
        person_path = os.path.join(DATASET_PATH, person)
        if not os.path.isdir(person_path):
            continue
        
        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)
            image_paths.append((img_path, person))
    
    # Process images
    print(f"Processing {len(image_paths)} images...")
    for img_path, person in image_paths:
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
    
    if embeddings:
        data = {"embeddings": embeddings, "names": names}
        os.makedirs("embeddings", exist_ok=True)
        with open(EMBEDDINGS_PATH, "wb") as f:
            pickle.dump(data, f)
        
        state.load_embeddings()
        return True, f"Generated {processed} embeddings ({failed} failed)", processed
    
    return False, "No valid faces found", 0


# ============================================================================
# ROUTES (Same as original)
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


# ============================================================================
# API ENDPOINTS (Same as original)
# ============================================================================

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
    
    with state.frame_lock:
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
    with state.frame_lock:
        if state.latest_annotated_frame is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"snapshot_{timestamp}.jpg"
            filepath = os.path.join(SNAPSHOTS_PATH, filename)
            cv2.imwrite(filepath, state.latest_annotated_frame)
            
            return jsonify({
                'success': True,
                'message': 'Snapshot saved',
                'filename': filename,
                'path': filepath
            })
    
    return jsonify({'success': False, 'message': 'No frame available'})

@app.route('/api/get_snapshot/<filename>')
def get_snapshot(filename):
    filepath = os.path.join(SNAPSHOTS_PATH, filename)
    if os.path.exists(filepath):
        return send_file(filepath, mimetype='image/jpeg')
    return jsonify({'error': 'Not found'}), 404

@app.route('/api/detection_status')
def detection_status():
    with state.frame_lock:
        return jsonify({
            'detection_info': state.detection_results,
            'current_command': state.current_command
        })

@app.route('/api/system_status')
def system_status():
    embeddings_exist = os.path.exists(EMBEDDINGS_PATH)
    embeddings_count = len(state.known_names) if state.known_names else 0
    
    dataset_persons = []
    if os.path.exists(DATASET_PATH):
        dataset_persons = [d for d in os.listdir(DATASET_PATH)
                          if os.path.isdir(os.path.join(DATASET_PATH, d))]
    
    return jsonify({
        'embeddings_loaded': embeddings_exist,
        'embeddings_count': embeddings_count,
        'dataset_persons': len(dataset_persons),
        'persons': dataset_persons,
        'camera_active': state.camera is not None,
        'current_command': state.current_command
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
                filename = f"{idx:04d}.jpg"
                filepath = os.path.join(person_dir, filename)
                image.save(filepath)
                saved_count += 1
        
        return jsonify({
            'success': True,
            'message': f'Saved {saved_count} images',
            'count': saved_count
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/delete_embeddings', methods=['DELETE'])
def delete_embeddings():
    try:
        if os.path.exists(EMBEDDINGS_PATH):
            os.remove(EMBEDDINGS_PATH)
            state.known_embeddings = None
            state.known_names = None
            state.normalized_embeddings = None
            return jsonify({
                'success': True,
                'message': 'Embeddings file deleted successfully'
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Embeddings file not found'
            })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error deleting embeddings: {str(e)}'
        })


# ============================================================================
# RUN
# ============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("OPTIMIZED FACE RECOGNITION SYSTEM")
    print("=" * 60)
    print(f"Camera Index: {CAMERA_INDEX}")
    print(f"Dataset: {DATASET_PATH}")
    print(f"Embeddings: {EMBEDDINGS_PATH}")
    print(f"Frame Skip: Every {DETECTION_SKIP_FRAMES + 1} frames")
    print(f"Detection Size: {DETECTION_SIZE}")
    print(f"Stream Quality: {STREAM_JPEG_QUALITY}%")
    print(f"Target FPS: {STREAM_FPS_TARGET}")
    print("=" * 60)
    
    # Initialize
    state.initialize_recognizer()
    state.load_embeddings()
    
    # Use production WSGI server for better performance
    try:
        from waitress import serve
        print("Using Waitress production server")
        serve(app, host='0.0.0.0', port=5000, threads=8)
    except ImportError:
        print("Waitress not found, using Flask dev server")
        print("Install waitress for better performance: pip install waitress")
        app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)
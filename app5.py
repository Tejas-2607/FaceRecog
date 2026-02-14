"""
Flask Web Application for Face Recognition Surveillance System
Based on original recognize_faces.py workflow
"""

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

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

app = Flask(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

# **IMPORTANT: Change camera index based on your device**
CAMERA_INDEX = 0  # ⚠️ CHANGE THIS ACCORDING TO YOUR DEVICE

# Paths
DATASET_PATH = "dataset"
EMBEDDINGS_PATH = "embeddings/face_embeddings.pkl"
SNAPSHOTS_PATH = "snapshots"

# Recognition settings
RECOGNITION_THRESHOLD = 0.65
MIN_DETECTION_CONFIDENCE = 0.5

# Create necessary directories
os.makedirs(DATASET_PATH, exist_ok=True)
os.makedirs("embeddings", exist_ok=True)
os.makedirs(SNAPSHOTS_PATH, exist_ok=True)

# ============================================================================
# GLOBAL STATE
# ============================================================================

class SystemState:
    """Manages global state for the recognition system"""
    
    def __init__(self):
        self.camera = None
        self.recognizer = None
        self.known_embeddings = None
        self.known_names = None
        self.command_parser = CommandParser()
        self.current_command = None
        self.frame_lock = threading.Lock()
        self.latest_frame = None
        self.detection_results = {}
        self.camera_lock = threading.Lock()
        
    def initialize_recognizer(self):
        """Load the face recognition model"""
        if self.recognizer is None:
            print("Loading InsightFace model...")
            self.recognizer = insightface.app.FaceAnalysis(
                name="buffalo_l",
                providers=["CPUExecutionProvider"]
            )
            self.recognizer.prepare(ctx_id=-1, det_size=(640, 640))
            print("✓ InsightFace model loaded")
    
    def load_embeddings(self):
        """Load face embeddings database"""
        if not os.path.exists(EMBEDDINGS_PATH):
            print("⚠️ No embeddings file found")
            return False
        
        try:
            with open(EMBEDDINGS_PATH, "rb") as f:
                data = pickle.load(f)
            
            self.known_embeddings = np.array(data["embeddings"])
            self.known_names = data["names"]
            print(f"✓ Loaded {len(self.known_names)} faces from database")
            return True
        except Exception as e:
            print(f"❌ Error loading embeddings: {e}")
            return False
    
    def get_camera(self):
        """Get or initialize camera"""
        with self.camera_lock:
            if self.camera is not None and self.camera.isOpened():
                return self.camera
            
            if self.camera is not None:
                self.camera.release()
            
            print(f"Opening camera {CAMERA_INDEX}...")
            self.camera = cv2.VideoCapture(CAMERA_INDEX)
            
            if not self.camera.isOpened():
                print(f"❌ Cannot open camera {CAMERA_INDEX}")
                return None
            
            # Set camera properties
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            print("✓ Camera opened successfully")
            return self.camera
    
    def release_camera(self):
        """Release camera resource"""
        with self.camera_lock:
            if self.camera is not None:
                self.camera.release()
                self.camera = None
                print("Camera released")


state = SystemState()


# ============================================================================
# RECOGNITION LOGIC (Based on original recognize_faces.py)
# ============================================================================

def recognize_and_draw_frame(frame, command_result=None):
    """
    Recognize faces and draw annotations on frame.
    Based on original recognize_faces.py logic.
    
    Args:
        frame: OpenCV frame
        command_result: Parsed command dictionary
    
    Returns:
        annotated_frame, detection_info
    """
    if state.recognizer is None:
        state.initialize_recognizer()
    
    if state.known_embeddings is None:
        state.load_embeddings()
    
    # If no embeddings loaded, show message
    if state.known_embeddings is None or len(state.known_embeddings) == 0:
        cv2.putText(frame, "No dataset/identity registered", (30, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return frame, {'total_faces': 0, 'message': 'No dataset'}
    
    # Detect faces
    faces = state.recognizer.get(frame)
    
    anchor_face = None
    detected_faces = []
    
    # ========== Recognition pass (EXACT copy from recognize_faces.py) ==========
    for face in faces:
        emb = face.embedding
        emb = emb / np.linalg.norm(emb)
        emb = emb.reshape(1, -1)
        
        sims = cosine_similarity(emb, state.known_embeddings)[0]
        best_idx = np.argmax(sims)
        best_score = sims[best_idx]
        
        if best_score > RECOGNITION_THRESHOLD:
            name = state.known_names[best_idx]
        else:
            name = "Unknown"
        
        bbox = face.bbox.astype(int)
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) // 2
        
        detected_faces.append({
            "name": name,
            "score": best_score,
            "bbox": bbox,
            "center_x": center_x
        })
        
        # Check if this is anchor
        if command_result and name == command_result.get('reference_person'):
            anchor_face = {
                "bbox": bbox,
                "center_x": center_x
            }
    
    # ========== Drawing logic (EXACT copy from recognize_faces.py) ==========
    anchor_name = command_result.get('reference_person') if command_result else None
    direction = command_result.get('direction') if command_result else None
    
    # Anchor NOT detected
    if command_result and anchor_face is None:
        cv2.putText(frame, f"{anchor_name} not detected", (30, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Still draw other faces
        for data in detected_faces:
            x1, y1, x2, y2 = data["bbox"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"{data['name']} ({data['score']:.2f})",
                       (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    
    # Anchor detected
    elif command_result and anchor_face is not None:
        ax1, ay1, ax2, ay2 = anchor_face["bbox"]
        anchor_center_x = anchor_face["center_x"]
        
        # Draw anchor (GREEN)
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
            
            # Skip anchor
            if name == anchor_name:
                continue
            
            other_faces_exist = True
            
            # Draw other faces (BLUE)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"{name} ({score:.2f})", (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            
            # Direction Logic (YOUR RIGHT / LEFT)
            if direction == "right" and face_center_x < anchor_center_x:
                person_found = True
                cv2.putText(frame, f"Person detected: {name}", (30, 80),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            
            elif direction == "left" and face_center_x > anchor_center_x:
                person_found = True
                cv2.putText(frame, f"Person detected: {name}", (30, 80),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        
        # Final Decision Messages
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
    
    # No command - just draw all faces
    else:
        for data in detected_faces:
            x1, y1, x2, y2 = data["bbox"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"{data['name']} ({data['score']:.2f})",
                       (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    
    # Detection info
    detection_info = {
        'total_faces': len(detected_faces),
        'anchor_detected': anchor_face is not None if command_result else False,
        'target_detected': False,  # Will be set by auto-snapshot logic
        'faces': [{'name': f['name'], 'score': float(f['score'])} for f in detected_faces]
    }
    
    return frame, detection_info


# ============================================================================
# VIDEO STREAMING
# ============================================================================

def generate_frames():
    """Generate frames for video streaming"""
    import time
    
    camera = state.get_camera()
    if camera is None:
        # Return error frame
        error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(error_frame, "Camera Error - Check CAMERA_INDEX", (50, 240),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        ret, buffer = cv2.imencode('.jpg', error_frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        return
    
    while True:
        try:
            ret, frame = camera.read()
            if not ret:
                time.sleep(0.1)
                continue
            
            # Mirror frame
            frame = cv2.flip(frame, 1)
            
            # Process with recognition
            annotated, info = recognize_and_draw_frame(frame, state.current_command)
            
            # Store latest
            with state.frame_lock:
                state.latest_frame = annotated
                state.detection_results = info
            
            # Encode
            ret, buffer = cv2.imencode('.jpg', annotated,
                                      [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ret:
                continue
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            
            time.sleep(0.033)  # ~30 FPS
            
        except GeneratorExit:
            break
        except Exception as e:
            print(f"Stream error: {e}")
            time.sleep(0.1)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def generate_embeddings_from_dataset():
    """Generate embeddings from dataset folder"""
    if not os.path.exists(DATASET_PATH) or not os.listdir(DATASET_PATH):
        return False, "Dataset folder is empty", 0
    
    model = insightface.app.FaceAnalysis(
        name="buffalo_l",
        providers=["CPUExecutionProvider"]
    )
    model.prepare(ctx_id=-1)
    
    embeddings = []
    names = []
    processed = 0
    failed = 0
    
    for person in os.listdir(DATASET_PATH):
        person_path = os.path.join(DATASET_PATH, person)
        if not os.path.isdir(person_path):
            continue
        
        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)
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


# ============================================================================
# API ENDPOINTS
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
    """Capture current frame from server camera for dataset capture"""
    import base64
    
    with state.frame_lock:
        if state.latest_frame is not None:
            # Encode frame to JPEG
            ret, buffer = cv2.imencode('.jpg', state.latest_frame,
                                      [cv2.IMWRITE_JPEG_QUALITY, 90])
            if ret:
                # Convert to base64
                img_base64 = base64.b64encode(buffer).decode('utf-8')
                return jsonify({
                    'success': True,
                    'image': img_base64
                })
    
    return jsonify({'success': False, 'message': 'No frame available'})


@app.route('/api/generate_embeddings', methods=['POST'])
def api_generate_embeddings():
    success, message, count = generate_embeddings_from_dataset()
    return jsonify({'success': success, 'message': message, 'count': count})


@app.route('/api/capture_snapshot', methods=['POST'])
def capture_snapshot():
    with state.frame_lock:
        if state.latest_frame is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"snapshot_{timestamp}.jpg"
            filepath = os.path.join(SNAPSHOTS_PATH, filename)
            cv2.imwrite(filepath, state.latest_frame)
            
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
    """Delete the embeddings .pkl file"""
    try:
        if os.path.exists(EMBEDDINGS_PATH):
            os.remove(EMBEDDINGS_PATH)
            # Clear loaded embeddings from state
            state.known_embeddings = None
            state.known_names = None
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
    print("FACE RECOGNITION SYSTEM")
    print("=" * 60)
    print(f"Camera Index: {CAMERA_INDEX}")
    print(f"Dataset: {DATASET_PATH}")
    print(f"Embeddings: {EMBEDDINGS_PATH}")
    print("=" * 60)
    
    # Initialize
    state.initialize_recognizer()
    state.load_embeddings()
    
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
"""
Flask Web Application for Face Recognition Surveillance System
Provides web interface for dataset capture, recognition, and command execution
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
import json

app = Flask(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

# **IMPORTANT: Change camera index based on your device**
# 0 = Default webcam
# 1 = External webcam
# "rtsp://username:password@ip:port/stream" = CCTV RTSP stream
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
        self.detection_results = []
        
    def initialize_recognizer(self):
        """Load the face recognition model"""
        if self.recognizer is None:
            self.recognizer = insightface.app.FaceAnalysis(
                name="buffalo_l",
                providers=["CPUExecutionProvider"]
            )
            self.recognizer.prepare(ctx_id=-1, det_size=(640, 640))
    
    def load_embeddings(self):
        """Load face embeddings database"""
        if not os.path.exists(EMBEDDINGS_PATH):
            return False
        
        try:
            with open(EMBEDDINGS_PATH, "rb") as f:
                data = pickle.load(f)
            
            self.known_embeddings = np.array(data["embeddings"])
            self.known_names = data["names"]
            return True
        except Exception as e:
            print(f"Error loading embeddings: {e}")
            return False
    
    def get_camera(self):
        """Get or initialize camera"""
        if self.camera is None or not self.camera.isOpened():
            self.camera = cv2.VideoCapture(CAMERA_INDEX)
        return self.camera
    
    def release_camera(self):
        """Release camera resource"""
        if self.camera is not None:
            self.camera.release()
            self.camera = None


state = SystemState()


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def generate_embeddings_from_dataset():
    """
    Generate embeddings from the dataset folder.
    Returns (success, message, count)
    """
    if not os.path.exists(DATASET_PATH) or not os.listdir(DATASET_PATH):
        return False, "Dataset folder is empty", 0
    
    # Load model
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
    
    # Save database
    if embeddings:
        data = {
            "embeddings": embeddings,
            "names": names
        }
        
        os.makedirs("embeddings", exist_ok=True)
        
        with open(EMBEDDINGS_PATH, "wb") as f:
            pickle.dump(data, f)
        
        # Reload into state
        state.load_embeddings()
        
        return True, f"Generated {processed} embeddings ({failed} failed)", processed
    else:
        return False, "No valid faces found in dataset", 0


def recognize_frame(frame, command_result=None):
    """
    Process a frame for face recognition and command execution.
    
    Args:
        frame: OpenCV frame
        command_result: Parsed command dictionary (optional)
    
    Returns:
        annotated_frame, detection_info
    """
    if state.recognizer is None:
        state.initialize_recognizer()
    
    if state.known_embeddings is None:
        state.load_embeddings()
    
    # Detect faces
    faces = state.recognizer.get(frame)
    
    anchor_face = None
    detected_faces = []
    
    # Recognition pass
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
        
        face_info = {
            "name": name,
            "score": best_score,
            "bbox": bbox,
            "center_x": center_x
        }
        
        detected_faces.append(face_info)
        
        # Check if this is the anchor person
        if command_result and name == command_result.get('reference_person'):
            anchor_face = face_info
    
    # Process command if provided
    message = None
    target_person = None
    
    if command_result and command_result.get('valid'):
        reference_person = command_result['reference_person']
        direction = command_result['direction']
        position = command_result['position']
        
        # Check if anchor detected
        if anchor_face is None:
            message = f"❌ {reference_person} not detected"
            color = (0, 0, 255)  # Red
        else:
            anchor_center = anchor_face['center_x']
            
            # Filter faces by direction
            if direction == 'right':
                # Right means face_center < anchor_center (in mirrored view)
                candidates = [f for f in detected_faces 
                            if f['name'] != reference_person and f['center_x'] < anchor_center]
                # Sort by distance from anchor
                candidates.sort(key=lambda f: anchor_center - f['center_x'])
            else:  # left
                # Left means face_center > anchor_center (in mirrored view)
                candidates = [f for f in detected_faces 
                            if f['name'] != reference_person and f['center_x'] > anchor_center]
                # Sort by distance from anchor
                candidates.sort(key=lambda f: f['center_x'] - anchor_center)
            
            # Check if we have enough candidates
            if len(candidates) >= position:
                target_person = candidates[position - 1]
                message = f"✓ {target_person['name']} detected"
                color = (0, 255, 0)  # Green
            elif len(candidates) == 0:
                message = f"❌ No person on {direction.upper()}"
                color = (0, 0, 255)  # Red
            else:
                message = f"❌ Only {len(candidates)} person(s) on {direction.upper()}"
                color = (0, 0, 255)  # Red
    
    # Draw annotations
    annotated = frame.copy()
    
    for face_data in detected_faces:
        x1, y1, x2, y2 = face_data['bbox']
        name = face_data['name']
        score = face_data['score']
        
        # Determine color
        if command_result and face_data == anchor_face:
            box_color = (0, 255, 0)  # Green for anchor
            label = f"{name} (Anchor)"
        elif command_result and target_person and face_data == target_person:
            box_color = (255, 0, 255)  # Magenta for target
            label = f"{name} (TARGET)"
        else:
            box_color = (255, 0, 0)  # Blue for others
            label = f"{name} ({score:.2f})"
        
        # Draw box and label
        cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)
        cv2.putText(annotated, label, (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
    
    # Draw message at top
    if message:
        cv2.putText(annotated, message, (30, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    
    detection_info = {
        'total_faces': len(detected_faces),
        'anchor_detected': anchor_face is not None,
        'target_detected': target_person is not None,
        'message': message,
        'faces': [{'name': f['name'], 'score': float(f['score'])} for f in detected_faces]
    }
    
    return annotated, detection_info


def generate_frames():
    """Generator for video streaming"""
    camera = state.get_camera()
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        # Mirror the frame
        frame = cv2.flip(frame, 1)
        
        # Process with current command
        annotated, info = recognize_frame(frame, state.current_command)
        
        # Store latest frame
        with state.frame_lock:
            state.latest_frame = annotated
            state.detection_results = info
        
        # Encode frame
        ret, buffer = cv2.imencode('.jpg', annotated)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


# ============================================================================
# ROUTES - Main Pages
# ============================================================================

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')


@app.route('/capture')
def capture_page():
    """Dataset capture page"""
    return render_template('capture.html')


@app.route('/recognize')
def recognize_page():
    """Recognition page"""
    return render_template('recognize.html')


@app.route('/manage')
def manage_page():
    """Dataset management page"""
    persons = []
    if os.path.exists(DATASET_PATH):
        for person in os.listdir(DATASET_PATH):
            person_path = os.path.join(DATASET_PATH, person)
            if os.path.isdir(person_path):
                count = len([f for f in os.listdir(person_path) if f.endswith('.jpg')])
                persons.append({'name': person, 'count': count})
    
    return render_template('manage.html', persons=persons)


# ============================================================================
# ROUTES - Video Streaming
# ============================================================================

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')


# ============================================================================
# ROUTES - API Endpoints
# ============================================================================

@app.route('/api/set_command', methods=['POST'])
def set_command():
    """Set the current recognition command"""
    data = request.json
    command_text = data.get('command', '')
    
    result = state.command_parser.parse(command_text)
    
    if result['valid']:
        state.current_command = result
        feedback = state.command_parser.format_feedback(result)
        return jsonify({
            'success': True,
            'message': feedback,
            'command': result
        })
    else:
        state.current_command = None
        return jsonify({
            'success': False,
            'message': f"Invalid command: {result['error']}",
            'command': result
        })


@app.route('/api/clear_command', methods=['POST'])
def clear_command():
    """Clear the current command"""
    state.current_command = None
    return jsonify({'success': True, 'message': 'Command cleared'})


@app.route('/api/generate_embeddings', methods=['POST'])
def api_generate_embeddings():
    """Generate embeddings from dataset"""
    success, message, count = generate_embeddings_from_dataset()
    return jsonify({
        'success': success,
        'message': message,
        'count': count
    })


@app.route('/api/capture_snapshot', methods=['POST'])
def capture_snapshot():
    """Capture current frame as snapshot"""
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


@app.route('/api/system_status')
def system_status():
    """Get current system status"""
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
        'camera_active': state.camera is not None and state.camera.isOpened(),
        'current_command': state.current_command
    })


@app.route('/api/delete_person/<person_name>', methods=['DELETE'])
def delete_person(person_name):
    """Delete a person from dataset"""
    person_path = os.path.join(DATASET_PATH, person_name)
    
    if os.path.exists(person_path):
        import shutil
        shutil.rmtree(person_path)
        return jsonify({'success': True, 'message': f'{person_name} deleted'})
    
    return jsonify({'success': False, 'message': 'Person not found'})


# ============================================================================
# CLEANUP
# ============================================================================

@app.teardown_appcontext
def cleanup(error=None):
    """Cleanup resources"""
    state.release_camera()


# ============================================================================
# RUN APPLICATION
# ============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("FACE RECOGNITION SURVEILLANCE SYSTEM")
    print("=" * 60)
    print(f"Camera Index: {CAMERA_INDEX}")
    print(f"Dataset Path: {DATASET_PATH}")
    print(f"Embeddings Path: {EMBEDDINGS_PATH}")
    print("=" * 60)
    print("\nStarting Flask server...")
    print("Access the application at: http://localhost:5000")
    print("=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)

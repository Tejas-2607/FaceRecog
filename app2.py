from flask import Flask, render_template, Response, request, jsonify
from flask_cors import CORS
import cv2
import insightface
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import threading
import time

app = Flask(__name__)
CORS(app)

# Global variables
current_command = {"anchor": "", "direction": None, "target_person": None, "active": False}
snapshot_data = {"image": None, "person_name": None, "position_info": None, "pending_validation": False}
recognition_lock = threading.Lock()

# Load embeddings database
try:
    with open("embeddings/face_embeddings.pkl", "rb") as f:
        data = pickle.load(f)
    known_embeddings = np.array(data["embeddings"])
    known_names = data["names"]
    print(f"✅ Loaded {len(known_names)} faces from database")
except FileNotFoundError:
    print("⚠️ No embeddings found. Please run generate_embeddings.py first")
    known_embeddings = np.array([])
    known_names = []

# Load InsightFace model
recognizer = insightface.app.FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)
recognizer.prepare(ctx_id=-1, det_size=(640, 640))

class VideoCamera:
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        if not self.video.isOpened():
            print("⚠️ Webcam 0 not available, trying webcam 1...")
            self.video = cv2.VideoCapture(1)
        
    def __del__(self):
        self.video.release()
    
    def parse_command(self, command):
        """Parse user command and extract direction"""
        command = command.lower()
        if "right" in command:
            return "right"
        elif "left" in command:
            return "left"
        return None
    
    def get_frame(self):
        success, frame = self.video.read()
        if not success:
            return None
            
        # Mirror the frame
        frame = cv2.flip(frame, 1)
        
        with recognition_lock:
            anchor_name = current_command.get("anchor", "")
            direction = current_command.get("direction")
            target_person = current_command.get("target_person")
            active = current_command.get("active", False)
        
        if not active:
            # Just show the video feed
            ret, jpeg = cv2.imencode('.jpg', frame)
            return jpeg.tobytes()
        
        # Run face recognition
        faces = recognizer.get(frame)
        
        anchor_face = None
        detected_faces = []
        target_found = False
        target_face_data = None
        
        # Recognition pass
        for face in faces:
            emb = face.embedding
            emb = emb / np.linalg.norm(emb)
            emb = emb.reshape(1, -1)
            
            if len(known_embeddings) > 0:
                sims = cosine_similarity(emb, known_embeddings)[0]
                best_idx = np.argmax(sims)
                best_score = sims[best_idx]
                
                if best_score > 0.65:
                    name = known_names[best_idx]
                else:
                    name = "Unknown"
            else:
                name = "Unknown"
                best_score = 0
            
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox
            center_x = (x1 + x2) // 2
            
            detected_faces.append({
                "name": name,
                "score": best_score,
                "bbox": bbox,
                "center_x": center_x
            })
            
            if name == anchor_name:
                anchor_face = {
                    "bbox": bbox,
                    "center_x": center_x
                }
            
            # Check for direct detection target
            if direction == "direct" and target_person and name.lower() == target_person.lower():
                target_found = True
                target_face_data = {
                    "name": name,
                    "bbox": bbox,
                    "score": best_score
                }
        
        # Handle direct person detection mode
        if direction == "direct":
            if target_found and target_face_data:
                x1, y1, x2, y2 = target_face_data["bbox"]
                # Draw bright yellow box for target person
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
                cv2.putText(frame, f"{target_face_data['name']} (TARGET)",
                           (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                           0.8, (0, 255, 255), 2)
                cv2.putText(frame, f"✓ {target_face_data['name']} detected",
                           (30, 80), cv2.FONT_HERSHEY_SIMPLEX,
                           1.2, (0, 255, 0), 3)
                
                # Store snapshot data
                with recognition_lock:
                    snapshot_data["image"] = frame.copy()
                    snapshot_data["person_name"] = target_face_data['name']
                    snapshot_data["position_info"] = "Direct detection"
                    snapshot_data["pending_validation"] = True
            else:
                cv2.putText(frame, f"✗ {target_person} not detected",
                           (30, 80), cv2.FONT_HERSHEY_SIMPLEX,
                           1.2, (0, 0, 255), 3)
            
            # Draw other faces in blue
            for data in detected_faces:
                if target_found and data["name"] == target_face_data["name"]:
                    continue
                x1, y1, x2, y2 = data["bbox"]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 100, 0), 2)
                cv2.putText(frame, f"{data['name']} ({data['score']:.2f})",
                           (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                           0.7, (255, 100, 0), 2)
        
        # Handle traditional left/right detection
        elif anchor_face is None:
            cv2.putText(frame, f"{anchor_name} not detected",
                       (30, 40), cv2.FONT_HERSHEY_SIMPLEX,
                       1, (0, 0, 255), 2)
        else:
            ax1, ay1, ax2, ay2 = anchor_face["bbox"]
            anchor_center_x = anchor_face["center_x"]
            
            # Draw anchor
            cv2.rectangle(frame, (ax1, ay1), (ax2, ay2), (0, 255, 0), 3)
            cv2.putText(frame, f"{anchor_name} (You)",
                       (ax1, ay1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                       0.8, (0, 255, 0), 2)
            
            person_found = False
            other_faces_exist = False
            found_person_data = None
            position_count = 0
            
            for data in detected_faces:
                name = data["name"]
                score = data["score"]
                x1, y1, x2, y2 = data["bbox"]
                face_center_x = data["center_x"]
                
                if name == anchor_name:
                    continue
                
                other_faces_exist = True
                
                # Draw other faces
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 100, 0), 2)
                cv2.putText(frame, f"{name} ({score:.2f})",
                           (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                           0.7, (255, 100, 0), 2)
                
                # Direction logic
                if direction == "right" and face_center_x < anchor_center_x:
                    person_found = True
                    position_count += 1
                    found_person_data = {
                        "name": name,
                        "bbox": (x1, y1, x2, y2),
                        "position": position_count
                    }
                    cv2.putText(frame, f"✓ Person found: {name}",
                               (30, 80), cv2.FONT_HERSHEY_SIMPLEX,
                               1.2, (0, 255, 0), 3)
                    
                    # Store snapshot data
                    with recognition_lock:
                        snapshot_data["image"] = frame.copy()
                        snapshot_data["person_name"] = name
                        snapshot_data["position_info"] = f"Position {position_count} on {direction} of {anchor_name}"
                        snapshot_data["pending_validation"] = True
                        
                elif direction == "left" and face_center_x > anchor_center_x:
                    person_found = True
                    position_count += 1
                    found_person_data = {
                        "name": name,
                        "bbox": (x1, y1, x2, y2),
                        "position": position_count
                    }
                    cv2.putText(frame, f"✓ Person found: {name}",
                               (30, 80), cv2.FONT_HERSHEY_SIMPLEX,
                               1.2, (0, 255, 0), 3)
                    
                    # Store snapshot data
                    with recognition_lock:
                        snapshot_data["image"] = frame.copy()
                        snapshot_data["person_name"] = name
                        snapshot_data["position_info"] = f"Position {position_count} on {direction} of {anchor_name}"
                        snapshot_data["pending_validation"] = True
            
            # Messages
            if not person_found:
                if not other_faces_exist:
                    cv2.putText(frame, "Move the camera",
                               (30, 80), cv2.FONT_HERSHEY_SIMPLEX,
                               1, (255, 150, 0), 2)
                else:
                    msg = f"No person on your {direction.upper()}"
                    cv2.putText(frame, msg,
                               (30, 80), cv2.FONT_HERSHEY_SIMPLEX,
                               1, (0, 0, 255), 2)
        
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()

def gen(camera):
    while True:
        frame = camera.get_frame()
        if frame is None:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/set_command', methods=['POST'])
def set_command():
    data = request.json
    anchor = data.get('anchor', '').strip()
    command = data.get('command', '').strip()
    
    direction = None
    target_person = None
    
    if command:
        command_lower = command.lower()
        
        # Check for direct person detection (e.g., "detect User1", "identify Alice")
        detect_words = ['detect', 'identify', 'scan', 'find', 'show', 'locate']
        is_direct_detection = any(word in command_lower for word in detect_words)
        
        # Check if command has left/right direction
        has_direction = "right" in command_lower or "left" in command_lower
        
        if is_direct_detection and not has_direction:
            # Direct person detection mode
            for word in detect_words:
                if word in command_lower:
                    # Extract target person name (everything after the detect word)
                    parts = command.split()
                    try:
                        idx = next(i for i, w in enumerate(parts) if w.lower() == word)
                        if idx + 1 < len(parts):
                            target_person = ' '.join(parts[idx + 1:]).strip()
                            break
                    except StopIteration:
                        pass
            
            direction = "direct"  # Special mode for direct detection
        else:
            # Traditional left/right detection
            if "right" in command_lower:
                direction = "right"
            elif "left" in command_lower:
                direction = "left"
    
    with recognition_lock:
        current_command["anchor"] = anchor
        current_command["direction"] = direction
        current_command["target_person"] = target_person
        current_command["active"] = bool((anchor and direction) or (direction == "direct" and target_person))
    
    return jsonify({
        "status": "success",
        "anchor": anchor,
        "direction": direction,
        "target_person": target_person,
        "active": current_command["active"]
    })

@app.route('/stop_detection', methods=['POST'])
def stop_detection():
    with recognition_lock:
        current_command["active"] = False
    return jsonify({"status": "stopped"})

@app.route('/get_known_faces', methods=['GET'])
def get_known_faces():
    return jsonify({
        "faces": list(set(known_names)),
        "count": len(known_names)
    })

@app.route('/get_snapshot', methods=['GET'])
def get_snapshot():
    """Get the current pending snapshot for validation"""
    with recognition_lock:
        if snapshot_data["pending_validation"] and snapshot_data["image"] is not None:
            # Encode image to base64
            _, buffer = cv2.imencode('.jpg', snapshot_data["image"])
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            
            return jsonify({
                "has_snapshot": True,
                "image": img_base64,
                "person_name": snapshot_data["person_name"],
                "position_info": snapshot_data["position_info"]
            })
        else:
            return jsonify({
                "has_snapshot": False
            })

@app.route('/save_snapshot', methods=['POST'])
def save_snapshot():
    """Save the validated snapshot with annotation"""
    data = request.json
    confirmed = data.get('confirmed', False)
    person_name = data.get('person_name', 'Unknown')
    
    if not confirmed:
        with recognition_lock:
            snapshot_data["pending_validation"] = False
            snapshot_data["image"] = None
        return jsonify({"success": True, "message": "Snapshot discarded"})
    
    with recognition_lock:
        if snapshot_data["image"] is None:
            return jsonify({"success": False, "message": "No snapshot available"})
        
        # Create snapshots directory if it doesn't exist
        import os
        os.makedirs("snapshots", exist_ok=True)
        
        # Generate filename with timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"snapshots/{person_name}_{timestamp}.jpg"
        
        # Save the annotated image
        cv2.imwrite(filename, snapshot_data["image"])
        
        # Clear snapshot data
        snapshot_data["pending_validation"] = False
        saved_image = snapshot_data["image"].copy()
        snapshot_data["image"] = None
        
    return jsonify({
        "success": True,
        "message": "Snapshot saved successfully",
        "filename": filename,
        "ask_sketch": True  # Flag to ask about sketch generation
    })

@app.route('/generate_sketch', methods=['POST'])
def generate_sketch():
    """Generate a sketch/schematic diagram of the person"""
    data = request.json
    image_path = data.get('image_path', '')
    
    if not image_path or not os.path.exists(image_path):
        return jsonify({"success": False, "message": "Image not found"})
    
    try:
        # Load the saved image
        img = cv2.imread(image_path)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply edge detection for sketch effect
        edges = cv2.Canny(gray, 50, 150)
        
        # Invert for sketch effect (black lines on white)
        sketch = cv2.bitwise_not(edges)
        
        # Optional: Add some smoothing
        sketch = cv2.GaussianBlur(sketch, (3, 3), 0)
        
        # Alternative: Pencil sketch effect
        gray_blur = cv2.GaussianBlur(gray, (21, 21), 0)
        sketch_alt = cv2.divide(gray, gray_blur, scale=256.0)
        
        # Generate filename for sketch
        base_name = os.path.splitext(image_path)[0]
        sketch_path = f"{base_name}_sketch.jpg"
        sketch_alt_path = f"{base_name}_pencil_sketch.jpg"
        
        # Save both sketch versions
        cv2.imwrite(sketch_path, sketch)
        cv2.imwrite(sketch_alt_path, sketch_alt)
        
        return jsonify({
            "success": True,
            "message": "Sketch generated successfully",
            "sketch_path": sketch_path,
            "pencil_sketch_path": sketch_alt_path
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error generating sketch: {str(e)}"
        })

if __name__ == '__main__':
    import base64
    import os
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)

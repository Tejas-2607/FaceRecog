"""
Face Dataset Capture Script
Captures face images for building the recognition database
"""

import cv2
import os
from insightface.app import FaceAnalysis

# ============================================================================
# CONFIGURATION
# ============================================================================

# **IMPORTANT: Change camera index based on your device**
# 0 = Default webcam (built-in laptop camera)
# 1 = External USB webcam
# 2+ = Additional cameras
# "rtsp://username:password@ip:port/stream" = CCTV RTSP stream
CAMERA_INDEX = 0  # ⚠️ CHANGE THIS ACCORDING TO YOUR DEVICE

# Capture settings
MAX_IMAGES = 50           # Number of images to capture per person
MIN_CONFIDENCE = 0.5      # Minimum face detection confidence
MARGIN = 0.4              # Face crop margin (0.4 = 40% padding around face)

# Paths
DATASET_BASE_PATH = "dataset"

# ============================================================================
# MAIN SCRIPT
# ============================================================================

def main():
    """Main capture function"""
    
    # Get person name
    person_name = input("Enter person name: ").strip()
    
    if not person_name:
        print("❌ Error: Person name cannot be empty!")
        return
    
    # Create save directory
    save_path = os.path.join(DATASET_BASE_PATH, person_name)
    os.makedirs(save_path, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"CAPTURING DATASET FOR: {person_name}")
    print(f"{'='*60}")
    print(f"Save path: {save_path}")
    print(f"Target images: {MAX_IMAGES}")
    print(f"Press 'q' to quit early")
    print(f"{'='*60}\n")
    
    # Initialize InsightFace (detection only - CPU safe)
    print("Loading face detection model...")
    app = FaceAnalysis(
        allowed_modules=['detection'],
        providers=['CPUExecutionProvider']
    )
    app.prepare(ctx_id=-1, det_size=(640, 640))
    print("✓ Model loaded\n")
    
    # Initialize camera
    print(f"Opening camera (index: {CAMERA_INDEX})...")
    cap = cv2.VideoCapture(CAMERA_INDEX)
    
    if not cap.isOpened():
        print("❌ Error: Could not open camera!")
        print("Try changing CAMERA_INDEX at the top of this file")
        return
    
    print("✓ Camera opened\n")
    print("Starting capture in 3 seconds...")
    print("Look at the camera and move your head slowly")
    print("to capture different angles...\n")
    
    # Wait a moment before starting
    cv2.waitKey(3000)
    
    count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Error: Could not read frame from camera")
            break
        
        # Mirror the frame for natural interaction
        frame = cv2.flip(frame, 1)
        
        # Detect faces
        faces = app.get(frame)
        
        for face in faces:
            # Skip low-confidence detections
            if face.det_score < MIN_CONFIDENCE:
                continue
            
            # Get bounding box
            x1, y1, x2, y2 = face.bbox.astype(int)
            
            h, w, _ = frame.shape
            bw = x2 - x1
            bh = y2 - y1
            
            # Expand bounding box with margin
            x1 = max(0, int(x1 - bw * MARGIN))
            y1 = max(0, int(y1 - bh * MARGIN))
            x2 = min(w, int(x2 + bw * MARGIN))
            y2 = min(h, int(y2 + bh * MARGIN))
            
            # Crop face region
            face_crop = frame[y1:y2, x1:x2]
            
            if face_crop.size == 0:
                continue
            
            # Save image
            file_name = os.path.join(save_path, f"{count:04d}.jpg")
            cv2.imwrite(file_name, face_crop)
            count += 1
            
            # Draw rectangle on frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Show progress
            progress_text = f"Captured: {count}/{MAX_IMAGES}"
            cv2.putText(frame, progress_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display frame
        cv2.imshow("Capture Faces - Press 'q' to quit", frame)
        
        # Check for quit or completion
        if cv2.waitKey(1) & 0xFF == ord('q') or count >= MAX_IMAGES:
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\n{'='*60}")
    print(f"✓ Capture complete!")
    print(f"Captured {count} images for {person_name}")
    print(f"Saved to: {save_path}")
    print(f"{'='*60}\n")
    print("Next steps:")
    print("1. Run the Flask app: python app.py")
    print("2. Go to the home page and click 'Generate Embeddings'")
    print("3. Start recognition!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Capture interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")

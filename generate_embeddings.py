"""
Face Embeddings Generation Script
Processes dataset images and generates face embeddings database
"""

import insightface
import cv2
import os
import pickle
import numpy as np

# ============================================================================
# CONFIGURATION
# ============================================================================

DATASET_PATH = "dataset"
EMBEDDINGS_OUTPUT_PATH = "embeddings/face_embeddings.pkl"

# ============================================================================
# MAIN SCRIPT
# ============================================================================

def main():
    """Generate embeddings from dataset"""
    
    print("\n" + "="*60)
    print("FACE EMBEDDINGS GENERATOR")
    print("="*60)
    
    # Check if dataset exists
    if not os.path.exists(DATASET_PATH):
        print(f"❌ Error: Dataset folder not found at '{DATASET_PATH}'")
        print("Please run capture_faces.py first to create datasets")
        return
    
    # Check if dataset has any persons
    persons = [d for d in os.listdir(DATASET_PATH) 
              if os.path.isdir(os.path.join(DATASET_PATH, d))]
    
    if not persons:
        print(f"❌ Error: No person folders found in '{DATASET_PATH}'")
        print("Please run capture_faces.py to create datasets")
        return
    
    print(f"\nFound {len(persons)} person(s) in dataset:")
    for person in persons:
        person_path = os.path.join(DATASET_PATH, person)
        img_count = len([f for f in os.listdir(person_path) if f.endswith('.jpg')])
        print(f"  - {person}: {img_count} images")
    
    print("\nLoading InsightFace model...")
    
    # Load model
    model = insightface.app.FaceAnalysis(
        name="buffalo_l",
        providers=["CPUExecutionProvider"]
    )
    model.prepare(ctx_id=-1)
    
    print("✓ Model loaded\n")
    print("Processing images...\n")
    
    embeddings = []
    names = []
    processed = 0
    failed = 0
    
    # Process each person's images
    for person in persons:
        person_path = os.path.join(DATASET_PATH, person)
        
        if not os.path.isdir(person_path):
            continue
        
        print(f"Processing {person}...")
        person_processed = 0
        person_failed = 0
        
        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)
            img = cv2.imread(img_path)
            
            if img is None:
                print(f"  ✗ Failed to read: {img_name}")
                failed += 1
                person_failed += 1
                continue
            
            # Detect faces and extract embeddings
            faces = model.get(img)
            
            if len(faces) > 0:
                # Use the first (and typically only) face
                emb = faces[0].embedding
                emb = emb / np.linalg.norm(emb)  # Normalize
                
                embeddings.append(emb)
                names.append(person)
                processed += 1
                person_processed += 1
            else:
                print(f"  ✗ No face detected: {img_name}")
                failed += 1
                person_failed += 1
        
        print(f"  ✓ Processed: {person_processed}, Failed: {person_failed}\n")
    
    # Save embeddings database
    if embeddings:
        print("Saving embeddings database...")
        
        data = {
            "embeddings": embeddings,
            "names": names
        }
        
        os.makedirs("embeddings", exist_ok=True)
        
        with open(EMBEDDINGS_OUTPUT_PATH, "wb") as f:
            pickle.dump(data, f)
        
        print(f"✓ Saved to: {EMBEDDINGS_OUTPUT_PATH}\n")
        print("="*60)
        print("SUMMARY")
        print("="*60)
        print(f"Total embeddings generated: {processed}")
        print(f"Failed images: {failed}")
        print(f"Success rate: {(processed/(processed+failed)*100):.1f}%")
        print("="*60)
        print("\nDatabase ready for recognition! ✅")
        print("\nNext step: Run the Flask app (python app.py)")
        print("="*60 + "\n")
    else:
        print("❌ Error: No valid face embeddings generated")
        print("Please check your dataset images")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

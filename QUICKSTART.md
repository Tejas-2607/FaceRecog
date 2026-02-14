# 🚀 QUICK START GUIDE

## 5-Minute Setup

### 1️⃣ Install Dependencies (1 minute)

```bash
pip install opencv-python insightface onnxruntime numpy scikit-learn flask
```

### 2️⃣ Configure Camera (30 seconds)

Open these files and update CAMERA_INDEX:

**capture_faces.py** (Line 18):
```python
CAMERA_INDEX = 0  # Change to 1 for external webcam
```

**app.py** (Line 35):
```python
CAMERA_INDEX = 0  # Change to 1 for external webcam
```

### 3️⃣ Capture Datasets (2 minutes per person)

```bash
python capture_faces.py
```

- Enter person name (e.g., "User1")
- Look at camera, move head slowly
- Wait for 50 images to be captured

Repeat for each person you want to recognize.

### 4️⃣ Start the Application (30 seconds)

```bash
python app.py
```

Open browser to: **http://localhost:5000**

### 5️⃣ Generate Embeddings (1 minute)

On the home page, click **"Generate Embeddings"** button.

### 6️⃣ Start Recognition! (Ready to use)

Go to **Recognition** page and enter commands like:

```
detect person right to User1
find second person on left of Alice
```

---

## 💡 Key Points to Remember

### ⚠️ Camera Configuration
**ALWAYS check and update CAMERA_INDEX before running:**
- `0` = Built-in webcam
- `1` = External USB webcam
- `2+` = Additional cameras

### ✅ Dataset Quality
- Capture **50+ images** per person
- Use **good lighting**
- Include **different angles**

### 🎯 Command Format
```
[action] [position] person [direction] of [reference_person]
```

Example: `detect second person left of User1`

### 🔄 After Making Changes
- Added/deleted people? → **Regenerate embeddings**
- Changed thresholds? → **Restart Flask app**

---

## 📱 Web Interface Pages

1. **Home** (`/`) - Overview and system status
2. **Capture** (`/capture`) - Dataset capture instructions
3. **Recognition** (`/recognize`) - Live recognition and commands
4. **Manage** (`/manage`) - View/delete datasets

---

## 🆘 Common Issues

### Camera won't open?
→ Change CAMERA_INDEX (try 0, 1, 2)

### Face not detected?
→ Improve lighting, face camera directly

### Low recognition accuracy?
→ Capture more images, regenerate embeddings

### Command not working?
→ Check name spelling (case-sensitive!)

---

## 📞 Need Help?

Check the full **README.md** for:
- Detailed troubleshooting
- API documentation
- Advanced configuration
- Project architecture

---

**You're ready to go! 🎉**

# ====================================================================
# FACE RECOGNITION SYSTEM - CONFIGURATION FILE
# ====================================================================
# This file contains all configurable parameters for the system.
# Copy this to config.py and customize as needed.

# ====================================================================
# CAMERA SETTINGS
# ====================================================================

# **CRITICAL: Change this based on your device**
# 
# Common values:
#   0 = Built-in laptop webcam
#   1 = External USB webcam
#   2, 3, ... = Additional cameras
#   "rtsp://username:password@ip:port/stream" = CCTV RTSP stream
#
# Examples:
#   CAMERA_INDEX = 0
#   CAMERA_INDEX = 1
#   CAMERA_INDEX = "rtsp://admin:password@192.168.1.100:554/stream1"

CAMERA_INDEX = 0  # ⚠️ CHANGE THIS ACCORDING TO YOUR DEVICE

# Camera resolution (optional - leave None for default)
CAMERA_WIDTH = None   # e.g., 1280
CAMERA_HEIGHT = None  # e.g., 720

# ====================================================================
# DATASET SETTINGS
# ====================================================================

# Maximum images to capture per person
MAX_IMAGES_PER_PERSON = 50

# Minimum face detection confidence (0.0 - 1.0)
# Higher = stricter detection, fewer false positives
MIN_DETECTION_CONFIDENCE = 0.5

# Face crop margin (0.0 - 1.0)
# How much padding around detected face
# 0.4 = 40% padding around face bounding box
FACE_CROP_MARGIN = 0.4

# ====================================================================
# RECOGNITION SETTINGS
# ====================================================================

# Recognition confidence threshold (0.0 - 1.0)
# Higher = stricter matching, fewer false matches
# Lower = more lenient, may get false matches
# Recommended: 0.60 - 0.70
RECOGNITION_THRESHOLD = 0.65

# Face detection size for recognition
# Larger = more accurate but slower
# Recommended: 640x640
DETECTION_SIZE = (640, 640)

# ====================================================================
# FILE PATHS
# ====================================================================

# Base dataset directory
DATASET_PATH = "dataset"

# Embeddings database file
EMBEDDINGS_PATH = "embeddings/face_embeddings.pkl"

# Snapshots save directory
SNAPSHOTS_PATH = "snapshots"

# ====================================================================
# WEB SERVER SETTINGS
# ====================================================================

# Flask server host
# '0.0.0.0' = accessible from network
# '127.0.0.1' = localhost only
FLASK_HOST = '0.0.0.0'

# Flask server port
FLASK_PORT = 5000

# Debug mode
# True = auto-reload on code changes, verbose errors
# False = production mode
FLASK_DEBUG = True

# ====================================================================
# INSIGHTFACE MODEL SETTINGS
# ====================================================================

# Model name (don't change unless you know what you're doing)
INSIGHTFACE_MODEL = "buffalo_l"

# Compute device
# 'CPUExecutionProvider' = CPU (compatible with all systems)
# 'CUDAExecutionProvider' = GPU (requires CUDA setup)
EXECUTION_PROVIDERS = ['CPUExecutionProvider']

# Context ID for InsightFace
# -1 = CPU
#  0 = GPU 0
#  1 = GPU 1, etc.
CTX_ID = -1

# ====================================================================
# COMMAND PARSING SETTINGS
# ====================================================================

# Enable verbose command parsing (for debugging)
VERBOSE_PARSING = False

# Default position if not specified in command
DEFAULT_POSITION = 1  # First person

# ====================================================================
# PERFORMANCE SETTINGS
# ====================================================================

# Video stream quality (JPEG compression)
# 0-100, higher = better quality, larger bandwidth
VIDEO_QUALITY = 85

# Frame skip for performance
# Process every Nth frame (1 = every frame)
FRAME_SKIP = 1

# Maximum frames per second
MAX_FPS = 30

# ====================================================================
# SECURITY SETTINGS
# ====================================================================

# Enable/disable person deletion via API
ALLOW_DELETE_PERSON = True

# Require confirmation for destructive operations
REQUIRE_CONFIRMATION = True

# ====================================================================
# LOGGING SETTINGS
# ====================================================================

# Enable system logging
ENABLE_LOGGING = True

# Log level
# DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_LEVEL = "INFO"

# Log file path
LOG_FILE = "system.log"

# ====================================================================
# ADVANCED SETTINGS
# ====================================================================

# Enable experimental features
EXPERIMENTAL_FEATURES = False

# Cache embeddings in memory
CACHE_EMBEDDINGS = True

# Auto-reload embeddings on file change
AUTO_RELOAD_EMBEDDINGS = False

# ====================================================================
# CCTV SPECIFIC SETTINGS (if using RTSP)
# ====================================================================

# RTSP connection timeout (seconds)
RTSP_TIMEOUT = 10

# RTSP reconnection attempts
RTSP_RECONNECT_ATTEMPTS = 3

# RTSP buffer size
RTSP_BUFFER_SIZE = 1

# ====================================================================
# END OF CONFIGURATION
# ====================================================================

# To use this configuration:
# 1. Copy this file to config.py
# 2. Update the values above
# 3. Import in your scripts:
#    from config import CAMERA_INDEX, RECOGNITION_THRESHOLD, etc.

"""
Project Configuration
"""

# Camera Settings
CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
WINDOW_NAME = "Unified Driver Monitoring Dashboard"

# Display Settings
SHOW_FPS = True

# ==========================
# AI Models
# ==========================

# YOLO Face Detection Model
YOLO_FACE_MODEL = "trained_models/face/face_yolov8n.pt"

# MediaPipe Face Landmarker Model
FACE_LANDMARK_MODEL = "trained_models/face/face_landmarker.task"
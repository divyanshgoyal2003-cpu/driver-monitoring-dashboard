"""
Project Configuration

Every tunable value lives here. Durations are in seconds rather than
frames, so detection behaves the same at 10 FPS as it does at 30 FPS.
"""

# ==========================
# Video Source
# ==========================

CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# ==========================
# Display
# ==========================

WINDOW_NAME = "Unified Driver Monitoring Dashboard"
SHOW_FPS = True
SHOW_LANDMARKS = True

# The camera frame is scaled to this height before the side panel is
# attached, so the panel layout is identical at any camera resolution.
DISPLAY_HEIGHT = 720
PANEL_WIDTH = 400

# ==========================
# AI Models
# ==========================

# YOLO Face Detection Model
YOLO_FACE_MODEL = "trained_models/face/face_yolov8n.pt"
FACE_DETECTION_CONFIDENCE = 0.4

# MediaPipe Face Landmarker Model
FACE_LANDMARK_MODEL = "trained_models/face/face_landmarker.task"

# ==========================
# Blink Detection
# ==========================

EAR_THRESHOLD = 0.23

# Shortest eye closure counted as a blink. Equivalent to the previous
# MIN_FRAMES = 2 at 30 FPS.
BLINK_MIN_DURATION = 0.06

# Closures this long are microsleeps, not blinks.
MICROSLEEP_DURATION = 1.5
BLINK_MAX_DURATION = MICROSLEEP_DURATION

# Blink rate is reported in blinks per minute over this window.
BLINK_RATE_WINDOW = 60.0

# ==========================
# Yawn Detection
# ==========================

# Tuned against pixel-space MAR. The previous value of 0.6 was tuned
# against normalized coordinates, where x and y are scaled by different
# amounts and the ratio therefore drifts with head roll.
# 0.6 * (720 / 1280) = 0.3375 preserves the old behaviour for a level
# head at 720p while staying correct at any resolution and roll angle.
MAR_THRESHOLD = 0.34

# A yawn holds the mouth wide open for well over a second; requiring
# 0.8 s rejects speech and short mouth movements.
YAWN_MIN_DURATION = 0.8

# "Recent yawns" are counted over this window (5 minutes).
YAWN_RATE_WINDOW = 300.0

# ==========================
# Head Pose
# ==========================

HEAD_YAW_THRESHOLD = 15.0
HEAD_PITCH_THRESHOLD = 15.0

# The driver's neutral pose is learned over the first few seconds the
# face is visible (press C to recalibrate). This removes the offset
# caused by the camera not sitting directly in front of the face.
HEAD_POSE_CALIBRATION_TIME = 3.0
HEAD_POSE_MAX_OFFSET = 25.0

# ==========================
# Eye Tracking (Gaze)
# ==========================

# The iris is modelled as sitting on a sphere whose radius is this
# fraction of the eye's corner-to-corner width (eyeball radius ~12 mm,
# eye opening ~30 mm). An iris offset of d eye-widths from the centre
# then means an eye rotation of asin(d / 0.4).
GAZE_EYE_RADIUS_RATIO = 0.4

# Combined head + eye yaw, in degrees, beyond which the gaze is LEFT or
# RIGHT.
GAZE_THRESHOLD = 15.0

# ==========================
# Drowsiness
# ==========================

# PERCLOS: the share of time the eyes were closed over the last minute.
# An alert driver sits around 5% from normal blinking.
PERCLOS_WINDOW = 60.0
PERCLOS_TIRED = 0.15
PERCLOS_DROWSY = 0.25

# PERCLOS is only judged once this much eye data has been collected.
PERCLOS_MIN_DATA = 10.0

# ==========================
# Distraction
# ==========================

# The gaze counts as off the road beyond these angles (degrees of
# combined head + eye yaw, or of head pitch up or down)...
OFF_ROAD_YAW = 25.0
OFF_ROAD_PITCH = 20.0

# ...and the driver is distracted once it stays off the road this long.
DISTRACTION_DURATION = 2.0

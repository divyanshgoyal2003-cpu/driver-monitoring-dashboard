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

# A driver's face is large in the frame, so face detection runs at 320 px
# instead of YOLO's default 640: the same faces, in about half the time.
FACE_DETECTION_IMAGE_SIZE = 320

# MediaPipe Face Landmarker Model
FACE_LANDMARK_MODEL = "trained_models/face/face_landmarker.task"

# YOLOv8n trained on COCO, used for its "cell phone" class
OBJECT_DETECTION_MODEL = "trained_models/objects/yolov8n.pt"

# FER+ facial expression model (ONNX, run with OpenCV's DNN module)
EMOTION_MODEL = "trained_models/emotion/emotion-ferplus-8.onnx"

# Importing ultralytics switches OpenCV's own threading off (to suit its
# training data loaders); it is switched back on with this many threads
# so the emotion model runs faster.
OPENCV_THREADS = 4

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

# ==========================
# Phone Detection
# ==========================

PHONE_CONFIDENCE = 0.35

# Object detection is the most expensive model, so it runs on every Nth
# frame and its result is reused in between.
PHONE_DETECTION_INTERVAL = 3

# A phone must be seen for this long before it counts as in use, and the
# usage ends once no phone has been seen for PHONE_RELEASE_TIME. This
# rides out single-frame false detections and missed frames.
PHONE_MIN_DURATION = 0.5
PHONE_RELEASE_TIME = 1.0

# ==========================
# Emotion Recognition
# ==========================

# The classifier runs on every Nth frame, staggered from phone detection
# so the two models do not land on the same frame.
EMOTION_INTERVAL = 3

# Square face crop, as a multiple of the face box's longer side. FER+
# was trained on tightly cropped faces.
EMOTION_FACE_MARGIN = 1.1

# Weight of each new prediction in the running average (0..1). Lower is
# steadier but slower to react.
EMOTION_SMOOTHING = 0.3

# Emotions that can impair driving. They are flagged once the smoothed
# confidence reaches EMOTION_ALERT_CONFIDENCE.
NEGATIVE_EMOTIONS = ("Angry", "Sad", "Fearful", "Disgusted")
EMOTION_ALERT_CONFIDENCE = 0.5

# ==========================
# Seatbelt Detection
# ==========================

# Runs on every Nth frame, staggered from phone and emotion detection.
SEATBELT_INTERVAL = 3

# The strap's edges must lean between these angles from horizontal. A
# shoulder-to-hip strap seen from the front sits at about 55-65 degrees;
# the wider range allows for off-centre cameras. Near-vertical edges
# (ties, shirt plackets) fall outside it.
SEATBELT_MIN_ANGLE = 30.0
SEATBELT_MAX_ANGLE = 70.0

# Strap width (distance between its two edges) as a fraction of the
# face width. A belt is ~48 mm; a face is ~150 mm.
SEATBELT_MIN_WIDTH = 0.15
SEATBELT_MAX_WIDTH = 0.55

# Per-frame results are pooled over this window: the belt is ON when it
# was found in at least ON_RATIO of the frames where the chest was
# visible, and OFF at or below OFF_RATIO (in between, the previous
# status holds).
SEATBELT_WINDOW = 3.0
SEATBELT_ON_RATIO = 0.5
SEATBELT_OFF_RATIO = 0.2

# ==========================
# Driver Safety Index
# ==========================

# Points taken off a perfect score of 100 while each condition is
# active. Conditions add up; the score never goes below 0.
RISK_PENALTIES = {
    "microsleep": 60,
    "phone": 35,
    "distracted": 30,
    "driver_absent": 30,
    "drowsy": 30,
    "tired": 15,
    "no_seatbelt": 15,
    "frequent_yawning": 10,
    "negative_emotion": 10,
}

# Score bands for the risk level: SAFE at or above SAFE_SCORE, CAUTION at
# or above CAUTION_SCORE, DANGER below.
SAFE_SCORE = 75
CAUTION_SCORE = 50

# The displayed score falls quickly when a risk appears and recovers
# slowly once it is gone (time constants in seconds).
SCORE_FALL_TIME = 0.5
SCORE_RISE_TIME = 5.0

# The driver counts as absent once no face has been seen for this long.
DRIVER_ABSENT_TIME = 2.0

# Yawns within YAWN_RATE_WINDOW that count as frequent yawning.
YAWN_ALERT_COUNT = 3

# ==========================
# Alerts
# ==========================

# Beep for critical and high-severity alerts (Windows only; press S to
# mute). The cooldown is the minimum gap between beeps, in seconds.
ENABLE_SOUND_ALERTS = True
CRITICAL_ALERT_COOLDOWN = 1.0
HIGH_ALERT_COOLDOWN = 3.0

# ==========================
# Session Logging
# ==========================

# Every run is saved here for the analytics dashboard (data/ is
# gitignored).
DATABASE_PATH = "data/driver_monitoring.db"

# A snapshot of the measurements is stored this often (seconds); events
# are stored as they happen. Writes are committed every
# DATABASE_COMMIT_INTERVAL seconds, so a crash loses at most that much.
METRICS_LOG_INTERVAL = 1.0
DATABASE_COMMIT_INTERVAL = 5.0

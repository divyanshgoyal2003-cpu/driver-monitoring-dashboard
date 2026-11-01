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
HAND_LANDMARK_MODEL = "trained_models/hands/hand_landmarker.task"

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
# Per-Driver EAR Calibration
# ==========================

# EAR_THRESHOLD above is a constant, and a constant assumes an eye
# shape. Eye aperture varies enough between people that one value
# cannot serve both ends: a driver whose open-eye EAR sits below it is
# read as permanently closed, and one well above it may never cross it
# even on a real blink. These settings learn the threshold per driver
# instead; set ADAPTIVE_EAR to False to go back to the constant.
ADAPTIVE_EAR = True

# Samples are collected over this long before the threshold is fixed.
# Long enough to contain a few natural blinks (people blink every 3-5 s)
# without making the driver wait.
EAR_CALIBRATION_TIME = 10.0

# The open and closed ends are taken as percentiles rather than the
# maximum and minimum, so one noisy frame cannot set the baseline.
EAR_OPEN_PERCENTILE = 85.0
EAR_CLOSED_PERCENTILE = 5.0

# Where the threshold sits within the driver's own range. 0.4 is nearer
# the closed end, since a blink passes through the middle quickly.
EAR_CLOSED_FRACTION = 0.4

# Used when no clear closure was seen during calibration: the threshold
# becomes this fraction of the open-eye baseline. This still removes the
# per-driver offset, but not the difference in range.
EAR_OPEN_FRACTION = 0.75

# The open and closed ends must differ by at least this much before the
# two-ended form is trusted.
EAR_MIN_RANGE = 0.08

EAR_MIN_SAMPLES = 20

# A learned threshold outside this range means calibration went wrong
# (a face half out of frame, say). It is clamped and the eye signal is
# trusted less.
EAR_THRESHOLD_MIN = 0.12
EAR_THRESHOLD_MAX = 0.32

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

# A face may go missing for this long without the off-road clock being
# abandoned. A head turned far enough to be a distraction is also a
# head the landmarker struggles to find, so short gaps happen in the
# middle of exactly the episodes worth catching, and starting the
# count again afterwards means a long turn never reports at all.
# Nothing is claimed while the face is missing; only the clock is kept.
DISTRACTION_FACE_GAP = 1.0

# Looking back at the road has to last this long to end an episode.
# Entering a distraction already requires a sustained reading; leaving
# one requires the same, because a head cannot swing fifty degrees and
# back inside a frame or two. A reading that says it did is a bad
# landmark fit, and letting one end the episode threw away whole turns.
DISTRACTION_RECOVERY = 0.3

# ==========================
# Phone Detection
# ==========================

PHONE_CONFIDENCE = 0.35

# Object detection is expensive, so it runs on every Nth frame and its
# result is reused in between.
#
# The three heavy models are spaced 6 apart rather than 3. At 3, with
# the offsets below, every single frame carried one of them and no
# frame was ever cheap. At 6 half the frames run landmarks alone, which
# measured 12.6 -> 18+ FPS. The eyes are sampled on every frame either
# way, so the extra frames go straight into blink detection, which is
# what the frame rate was starving.
PHONE_DETECTION_INTERVAL = 6

# A phone must be seen for this long before it counts as in use, and the
# usage ends once no phone has been seen for PHONE_RELEASE_TIME. This
# rides out single-frame false detections and missed frames.
PHONE_MIN_DURATION = 0.5
PHONE_RELEASE_TIME = 1.0

# ==========================
# Emotion Recognition
# ==========================

# The classifier runs on every Nth frame, staggered from phone detection
# so the two models do not land on the same frame. FER+ is the most
# expensive model in the pipeline by some way (measured ~67 ms against
# ~16 ms for the landmarks) while carrying the smallest penalty, so it
# is the first thing to run less often. Its output is smoothed by
# EmotionAnalyzer anyway, so a slower update is barely visible.
EMOTION_INTERVAL = 6

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
# With all three at 6 they land on frames 0, 3 and 5 of each group,
# leaving the rest free.
SEATBELT_INTERVAL = 6

# ==========================
# Hands on the Wheel
# ==========================

# Hand tracking used as a monitored signal, not as a control. Reaching
# toward the camera to operate the system would take a hand off the
# wheel and the eyes off the road, which is the opposite of the point.
#
# Runs on frame 2 of each group of six, the last slot the heavy models
# leave free. Hands are slow on the scale that matters here - the
# alarm needs HANDS_OFF_TIME of evidence - so sampling a few times a
# second is ample.
HAND_DETECTION_INTERVAL = 6
HAND_CONFIDENCE = 0.5

# How long with no hands in frame before it counts. Long enough to sit
# out a gear change or a reach for the indicator stalk.
HANDS_OFF_TIME = 3.0

# Whether an absence of hands means anything at all.
#
# This camera is aimed at a face, and on most setups the wheel is below
# the bottom edge - so "no hands found" is the normal state, not a
# warning. The monitor therefore has to earn the right to complain: it
# only judges once it has found hands HANDS_SEEN_SAMPLES times within
# the last HANDS_SEEN_WINDOW seconds, which is evidence the wheel is in
# shot. Until then the status is UNKNOWN. The window, rather than a
# running total, means a camera knocked out of position stops claiming
# it can see.
HANDS_SEEN_WINDOW = 60.0
HANDS_SEEN_SAMPLES = 10

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
# Camera Obstruction
# ==========================

# A covered lens and an empty seat both leave the monitor with no face,
# but they are not the same problem: one is the driver gone, the other
# is the system blinded, and in a fleet it is how the system gets
# defeated. They are told apart by variance. A dark cabin still has
# shapes, highlights and sensor noise in it; a covered lens is flat.
CAMERA_BLOCKED_BRIGHTNESS = 15.0
CAMERA_BLOCKED_VARIANCE = 8.0

# Both must hold for this long, so the first frames of a camera that
# has not woken up yet do not raise it.
CAMERA_BLOCKED_TIME = 2.0

# ==========================
# Low Light Enhancement
# ==========================

# CLAHE on the luma channel when the cabin is dim. This cannot conjure
# signal that the sensor never captured - no software can, and infrared
# needs hardware - but it does redistribute what little contrast there
# is, which sometimes brings a face back within MediaPipe's reach.
#
# It is off above LOW_LIGHT_BRIGHTNESS because it costs a full-frame
# colour conversion (~10 ms at 720p), and it never touches the
# brightness the risk engine reads: enhancement does not make it more
# trustworthy, so the confidence layer must still see the real light
# level. Pass --no-enhance to measure with it off.
LOW_LIGHT_ENHANCE = True
LOW_LIGHT_BRIGHTNESS = 70.0
LOW_LIGHT_CLIP_LIMIT = 3.0
LOW_LIGHT_GRID = 8

# ==========================
# Driver Safety Index
# ==========================

# Points taken off a perfect score of 100 while each condition is
# active. Conditions add up; the score never goes below 0.
RISK_PENALTIES = {
    "microsleep": 60,
    "phone": 35,
    "distracted": 30,
    "camera_blocked": 30,
    "hands_off": 25,
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
# Dynamic Risk Weighting
# ==========================

# RISK_PENALTIES above is what a condition costs at its worst. The three
# layers below scale that per frame:
#
#     penalty = base * severity * confidence * context
#
# severity  - how bad this instance is (a 6 s glance away vs a 2 s one)
# confidence- how much the detector that found it can be trusted
# context   - how much the situation amplifies it
#
# Severity and confidence can only ever reduce a penalty; context can
# only ever raise it. So `base` stays the documented maximum any single
# condition can cost, and the 100-point budget keeps its meaning.
#
# Set to False (or pass --static-weights) to score with the flat
# penalties instead. The two modes are kept side by side so the same
# recording can be run through both.
DYNAMIC_WEIGHTS = True

# ----- severity -----
# Each pair is (value at which the condition starts, value at which it
# reaches full weight). Below the first the penalty is scaled by
# SEVERITY_FLOOR; above the second it is unscaled.
SEVERITY_FLOOR = 0.4

# Eyes closed, in seconds, measured from the microsleep threshold.
SEVERITY_MICROSLEEP = (MICROSLEEP_DURATION, 4.0)

# PERCLOS, from each state's own threshold upward.
SEVERITY_DROWSY = (PERCLOS_DROWSY, 0.45)
SEVERITY_TIRED = (PERCLOS_TIRED, PERCLOS_DROWSY)

# Gaze off the road, in seconds.
SEVERITY_DISTRACTED = (DISTRACTION_DURATION, 6.0)
SEVERITY_HANDS_OFF = (HANDS_OFF_TIME, 10.0)

# Phone held, in seconds.
SEVERITY_PHONE = (PHONE_MIN_DURATION, 3.0)

# Yawns inside YAWN_RATE_WINDOW.
SEVERITY_YAWNING = (YAWN_ALERT_COUNT, 8)

# Face missing, in seconds.
SEVERITY_ABSENT = (DRIVER_ABSENT_TIME, 8.0)

# ----- confidence -----
# Never zero: a condition that is active still shows its alert at full
# strength, and only its contribution to the score is scaled.
CONFIDENCE_FLOOR = 0.3

# The seatbelt detector is a heuristic, and its OFF verdict is weakest
# right at the boundary. Pooled ratio at SEATBELT_OFF_RATIO scales to
# this; a ratio of 0 (no strap seen at all) scales to 1.
CONFIDENCE_SEATBELT_AT_BOUNDARY = 0.4

# (weakest reading, fully trusted reading, weight at the weakest end).
# Smoothed emotion confidence, from the alert threshold upward.
CONFIDENCE_EMOTION = (EMOTION_ALERT_CONFIDENCE, 0.9, 0.4)

# YOLO box confidence for a phone.
CONFIDENCE_PHONE = (PHONE_CONFIDENCE, 0.8, 0.5)

# ----- context -----
# Mean frame luminance (0-255) at which the cabin counts as dark, and at
# which it counts as well lit.
BRIGHTNESS_DARK = 40.0
BRIGHTNESS_LIT = 100.0

# Eye measurements get noisy in a dark cabin, so the eye-based
# conditions are trusted less there. This is deliberately the opposite
# push to the circadian multiplier below: risk rises at night while the
# ability to measure it falls, and an uncertain night-time reading
# should not hit harder than a confident daytime one.
CONFIDENCE_EYES_DARK = 0.55

# Drowsiness-related crash risk peaks in the small hours, with a
# secondary dip after lunch. Applied to the fatigue conditions only.
CIRCADIAN_MULTIPLIERS = (
    # (start hour, end hour, multiplier)
    (2, 6, 1.40),
    (0, 2, 1.20),
    (6, 7, 1.20),
    (14, 16, 1.15),
)

# Time at the wheel, applied to the fatigue conditions: reaches
# 1 + TIME_ON_TASK_MAX after TIME_ON_TASK_HOURS.
TIME_ON_TASK_HOURS = 2.0
TIME_ON_TASK_MAX = 0.25

# Poor visibility leaves less margin for a glance away, so the attention
# conditions are amplified by up to this much in the dark.
LOW_LIGHT_ATTENTION_MAX = 0.15

# Weather, applied to the fatigue AND the attention conditions.
#
# This is not a claim that rain makes a driver sleepier or more easily
# distracted. It is about what a lapse costs: a wet road lengthens the
# stopping distance and poor visibility shortens the warning, so the
# same two-second glance away leaves less room to recover from - and so
# does the same microsleep. The mechanism is identical for both
# families, which is why there is one table here and not two.
#
# Ordered by how much margin the road and the air leave: wet grip with
# some visibility lost in rain, dry grip with almost no visibility in
# fog, both poor in snow, the worst grip of all on ice. Crash-risk
# studies put adverse weather at roughly 1.5-2x clear conditions and
# ice far above that, but these are penalty weights inside a 100-point
# budget rather than crash odds, and this layer is deliberately
# conservative - see the README.
#
# Anything not listed weighs 1.0, so an unrecognised string cannot
# quietly inflate the score.
WEATHER_MULTIPLIERS = {
    "clear": 1.00,
    "cloudy": 1.00,
    "rain": 1.15,
    "heavy_rain": 1.25,
    "fog": 1.25,
    "snow": 1.30,
    "ice": 1.35,
}

# ==========================
# Weather Forecast
# ==========================

# Where the weather comes from when the driver does not say.
#
# Open-Meteo is free and needs no key. It answers for a POINT, not for
# the road ahead: city-resolution, up to 15 minutes old, and silent
# about the state of the tarmac. That is why the driver can always
# overrule it with W, and why every reading is logged with its source -
# a declared condition and a forecast one are not the same evidence.
#
# Set your coordinates to switch it on. Nothing but these numbers
# leaves the machine, and nothing happens at all while they are None.
WEATHER_LATITUDE = None
WEATHER_LONGITUDE = None

WEATHER_URL = "https://api.open-meteo.com/v1/forecast"

# The API's own `current` block refreshes every 900 s, so asking more
# often than that returns the same answer.
WEATHER_REFRESH = 900.0
WEATHER_TIMEOUT = 6.0

# Two refinements on top of the reported code, because the code alone
# can understate what the road is like.
#
# LIQUID rain at or below freezing is ice on the tarmac whatever the
# forecast calls it, and visibility this low is fog-grade however clear
# the sky is said to be. Both only ever pick the WORSE of the two
# readings. Snow is exempt from the first: it lands already frozen and
# grips better than a freezing-rain glaze.
ICE_TEMPERATURE = 0.5
FOG_VISIBILITY = 1000.0

# WMO weather codes, as Open-Meteo reports them, mapped onto the seven
# conditions above. A code that is not listed is treated as no reading
# at all rather than as "clear": a forecast the system cannot interpret
# must not silently remove the amplification.
WMO_WEATHER = {
    0: "clear",
    1: "clear",
    2: "cloudy",
    3: "cloudy",
    45: "fog",
    48: "fog",
    51: "rain",         # drizzle
    53: "rain",
    55: "rain",
    56: "ice",          # freezing drizzle
    57: "ice",
    61: "rain",
    63: "rain",
    65: "heavy_rain",
    66: "ice",          # freezing rain
    67: "ice",
    71: "snow",
    73: "snow",
    75: "snow",
    77: "snow",         # snow grains
    80: "rain",         # showers
    81: "rain",
    82: "heavy_rain",
    85: "snow",
    86: "snow",
    95: "heavy_rain",   # thunderstorm
    96: "heavy_rain",
    99: "heavy_rain",
}

# Which conditions each context multiplier applies to.
FATIGUE_CONDITIONS = ("microsleep", "drowsy", "tired", "frequent_yawning")
ATTENTION_CONDITIONS = ("distracted", "phone", "hands_off")

# Conditions whose measurement depends on readable eyes.
EYE_CONDITIONS = ("microsleep", "drowsy", "tired")

# ==========================
# Alerts
# ==========================

# Beep for critical and high-severity alerts (Windows only; press S to
# mute). The cooldown is the minimum gap between beeps, in seconds.
ENABLE_SOUND_ALERTS = True

# Alert patterns: a sequence of (frequency Hz, duration ms, gap after
# ms), played in order, so a tone can pulse and climb instead of
# sitting on one note.
#
# A single steady beep is the easy thing to tune out, and the easy
# thing to sleep through - the wrong property for the one alert that
# has to reach someone whose eyes have already closed. Both patterns
# pulse and rise, because interruption and rising pitch are what read
# as urgent; the rousing one climbs further, ends on a long note and
# says the whole thing twice.
#
# ROUSE goes to the fatigue conditions, NOTIFY to everything else. A
# drowsy driver and a driver holding a phone are both "high" severity,
# but only one of them has to be woken up.
ALERT_TONE_ROUSE = (
    (1200, 160, 70),
    (1600, 160, 70),
    (2100, 420, 170),
    (1200, 160, 70),
    (1600, 160, 70),
    (2100, 520, 0),
)

ALERT_TONE_NOTIFY = (
    (900, 140, 80),
    (1200, 260, 0),
)

# Minimum gap between the STARTS of two alerts, in seconds. Each has
# to clear its own pattern (ROUSE runs about 2.0 s, NOTIFY about 0.5 s)
# or the alert would simply run back to back: the silence between is
# what stops it becoming background noise.
CRITICAL_ALERT_COOLDOWN = 3.0
HIGH_ALERT_COOLDOWN = 4.0

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

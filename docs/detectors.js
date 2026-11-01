/*
 * The analytics layer, ported from analytics/ and risk_engine/.
 *
 * Every threshold here is the value from app/config.py, named the same
 * way, so the two implementations can be compared line by line. Where
 * the Python carries a comment explaining why a number is what it is,
 * that comment is carried over too -- a constant without its reason is
 * a constant nobody will dare change.
 *
 * One detector is NOT a faithful port, and says so where it is defined:
 * ExpressionAnalyzer reads MediaPipe's face blendshapes instead of
 * running FER+. The reasoning is in its own comment.
 */

import { grayscale, equalize, blur, medianForThresholds, canny, segments, findStrap }
  from "./vision.js";

export const CONFIG = {
  // --- eyes ---
  EAR_THRESHOLD: 0.23,
  BLINK_MIN_DURATION: 0.06,
  MICROSLEEP_DURATION: 1.5,
  BLINK_RATE_WINDOW: 60.0,

  EAR_CALIBRATION_TIME: 10.0,
  EAR_OPEN_PERCENTILE: 85.0,
  EAR_CLOSED_PERCENTILE: 5.0,
  EAR_CLOSED_FRACTION: 0.4,
  EAR_MIN_RANGE: 0.08,
  EAR_MIN_SAMPLES: 20,
  EAR_THRESHOLD_MIN: 0.12,
  EAR_THRESHOLD_MAX: 0.32,

  // --- PERCLOS ---
  PERCLOS_WINDOW: 60.0,
  PERCLOS_TIRED: 0.15,
  PERCLOS_DROWSY: 0.25,
  PERCLOS_MIN_DATA: 10.0,

  // --- mouth ---
  MAR_THRESHOLD: 0.34,
  YAWN_MIN_DURATION: 0.8,
  YAWN_RATE_WINDOW: 300.0,
  YAWN_ALERT_COUNT: 3,

  // --- head ---
  HEAD_POSE_CALIBRATION_TIME: 3.0,
  HEAD_POSE_MAX_OFFSET: 25.0,
  HEAD_POSE_MAX_MAR: 0.45,

  // --- distraction ---
  OFF_ROAD_YAW: 25.0,
  OFF_ROAD_PITCH: 20.0,
  DISTRACTION_DURATION: 2.0,
  DISTRACTION_FACE_GAP: 1.0,
  DISTRACTION_RECOVERY: 0.3,

  // --- absent ---
  DRIVER_ABSENT_TIME: 2.0,

  // --- camera obstruction ---
  // A covered lens and an empty seat both leave the monitor with no
  // face, but they are not the same problem: one is the driver gone,
  // the other is the system blinded, and in a fleet it is how the
  // system gets defeated. They are told apart by variance -- a dark
  // cabin still has shapes, highlights and sensor noise in it; a
  // covered lens is flat.
  CAMERA_BLOCKED_BRIGHTNESS: 15.0,
  CAMERA_BLOCKED_VARIANCE: 8.0,
  CAMERA_BLOCKED_TIME: 2.0,

  // Shorter than the dark-lens window because the evidence is far
  // stronger: a live sensor's noise means two consecutive frames are
  // never bit identical, so even one pair is already strange.
  CAMERA_FROZEN_TIME: 1.0,

  // --- phone ---
  PHONE_CONFIDENCE: 0.35,
  PHONE_DETECTION_INTERVAL: 6,
  PHONE_MIN_DURATION: 0.5,
  PHONE_RELEASE_TIME: 1.0,

  // --- hands on the wheel ---
  HAND_DETECTION_INTERVAL: 6,
  HAND_CONFIDENCE: 0.5,
  HANDS_OFF_TIME: 3.0,
  HANDS_SEEN_WINDOW: 60.0,
  HANDS_SEEN_SHARE: 0.3,
  HANDS_SEEN_WARMUP: 30.0,

  // --- seatbelt ---
  SEATBELT_INTERVAL: 6,
  SEATBELT_MIN_ANGLE: 30.0,
  SEATBELT_MAX_ANGLE: 70.0,
  SEATBELT_MIN_WIDTH: 0.15,
  SEATBELT_MAX_WIDTH: 0.55,
  SEATBELT_WINDOW: 3.0,
  SEATBELT_ON_RATIO: 0.5,
  SEATBELT_OFF_RATIO: 0.2,

  // --- expression ---
  // No interval: blendshapes come out of the same forward pass as the
  // landmarks, so running them "less often" would save nothing.
  EMOTION_SMOOTHING: 0.3,
  EMOTION_ALERT_CONFIDENCE: 0.5,

  // --- scoring ---
  SAFE_SCORE: 75,
  CAUTION_SCORE: 50,
  SCORE_FALL_TIME: 0.5,
  SCORE_RISE_TIME: 5.0,

  // --- the three weighting layers ---
  SEVERITY_FLOOR: 0.4,
  CONFIDENCE_FLOOR: 0.3,
  CONFIDENCE_SEATBELT_AT_BOUNDARY: 0.4,
  CONFIDENCE_EYES_DARK: 0.55,

  BRIGHTNESS_DARK: 40.0,
  BRIGHTNESS_LIT: 100.0,

  TIME_ON_TASK_HOURS: 2.0,
  TIME_ON_TASK_MAX: 0.25,
  LOW_LIGHT_ATTENTION_MAX: 0.15
};

// (weakest reading, fully trusted reading, weight at the weakest end).
const CONFIDENCE_EMOTION = [CONFIG.EMOTION_ALERT_CONFIDENCE, 0.9, 0.4];
const CONFIDENCE_PHONE = [CONFIG.PHONE_CONFIDENCE, 0.8, 0.5];

// Points taken off a perfect score of 100 while each condition is
// active. Conditions add up; the score never goes below 0.
export const RISK_PENALTIES = {
  microsleep: 60,
  phone: 35,
  distracted: 30,
  camera_blocked: 30,
  driver_absent: 30,
  drowsy: 30,
  hands_off: 25,
  tired: 15,
  no_seatbelt: 15,
  frequent_yawning: 10,
  negative_emotion: 10
};

// (condition, message, severity) most urgent first. The first active
// condition is the one shown in the banner.
export { SPOKEN };

export const CONDITIONS = [
  ["microsleep",       "MICROSLEEP - WAKE UP",   "critical"],
  ["phone",            "PHONE IN USE",           "high"],
  ["distracted",       "EYES OFF ROAD",          "high"],
  ["hands_off",        "HANDS OFF WHEEL",        "high"],
  ["camera_blocked",   "CAMERA BLOCKED",         "high"],
  ["driver_absent",    "DRIVER NOT VISIBLE",     "high"],
  ["drowsy",           "DROWSINESS DETECTED",    "high"],
  ["tired",            "SIGNS OF FATIGUE",       "medium"],
  ["no_seatbelt",      "SEATBELT NOT DETECTED",  "medium"],
  ["frequent_yawning", "FREQUENT YAWNING",       "medium"],
  ["negative_emotion", "NEGATIVE EXPRESSION",    "low"]
];

// What each condition SAYS, as opposed to what it shows.
//
// The banner is upper case because it has to be read at a glance from
// a mount, and handing that same string to a speech engine is how this
// ended up spelling "MICROSLEEP" and "UP" out letter by letter -- all
// caps is how you write an acronym, so that is what it was heard as.
//
// Natural case fixes the pronunciation, and once these are separate
// strings they can also stop being labels and start being
// instructions. A driver does not need to be told the name of the
// condition they are in; they need to be told the one thing that ends
// it. So the spoken line is an imperative wherever there is an action
// to name, and the shortest one that carries -- there is no point in a
// sentence still being read out after the moment it was about.
const SPOKEN = {
  microsleep:       "Wake up",
  phone:            "Put the phone down",
  distracted:       "Eyes on the road",
  hands_off:        "Hands on the wheel",
  camera_blocked:   "The camera is covered",
  driver_absent:    "Driver not visible",
  drowsy:           "You are getting drowsy. Take a break",
  tired:            "You are showing signs of fatigue",
  no_seatbelt:      "Check your seatbelt",
  frequent_yawning: "You have been yawning. Think about a break",
  negative_emotion: "You seem tense"
};

// (value at which the condition starts, value at which it reaches full
// weight). Below the first the penalty is scaled by SEVERITY_FLOOR.
const SEVERITY_SPANS = {
  microsleep:       [CONFIG.MICROSLEEP_DURATION, 4.0],
  drowsy:           [CONFIG.PERCLOS_DROWSY, 0.45],
  tired:            [CONFIG.PERCLOS_TIRED, CONFIG.PERCLOS_DROWSY],
  distracted:       [CONFIG.DISTRACTION_DURATION, 6.0],
  phone:            [CONFIG.PHONE_MIN_DURATION, 3.0],
  hands_off:        [CONFIG.HANDS_OFF_TIME, 10.0],
  frequent_yawning: [CONFIG.YAWN_ALERT_COUNT, 8],
  driver_absent:    [CONFIG.DRIVER_ABSENT_TIME, 8.0],
  camera_blocked:   [CONFIG.CAMERA_BLOCKED_TIME, 10.0]
};

// Hours when fatigue lapses are most likely to be real, applied to the
// fatigue conditions only. Follows published work on when fatigue
// crashes happen; see the README.
// Each band carries a name as well as a number, because a panel that
// shows only "x1.40" is asking the driver to take the clock on faith.
// The hour comes off their own device; saying which band it landed in
// is how that becomes visible rather than merely true.
const CIRCADIAN = [
  [2, 6, 1.40, "small hours"],
  [0, 2, 1.20, "after midnight"],
  [6, 7, 1.20, "dawn"],
  [14, 16, 1.15, "after lunch"]
];

/** The band an hour falls in, named. Always returns something, so the
 *  panel has a row to draw even when nothing is amplifying. */
export function circadianBand(hour) {
  for (const [start, end, multiplier, label] of CIRCADIAN) {
    if (hour >= start && hour < end) {
      return { multiplier, label, start, end };
    }
  }
  return { multiplier: 1.0, label: "ordinary hours", start: null, end: null };
}

// Which conditions each context multiplier applies to.
const FATIGUE = new Set(["microsleep", "drowsy", "tired", "frequent_yawning"]);
const ATTENTION = new Set(["distracted", "phone", "hands_off"]);

// Conditions whose measurement depends on readable eyes.
const EYE_CONDITIONS = new Set(["microsleep", "drowsy", "tired"]);

// MediaPipe Face Mesh indices, the same ones the Python uses.
const LEFT_EYE  = [33, 160, 158, 133, 153, 144];
const RIGHT_EYE = [362, 385, 387, 263, 373, 380];
const MOUTH_LEFT = 61, MOUTH_RIGHT = 291, MOUTH_TOP = 13, MOUTH_BOTTOM = 14;

// ----------------------------------------------------------------------
// helpers
// ----------------------------------------------------------------------

const clamp = (v, lo, hi) => Math.min(Math.max(v, lo), hi);

export function ramp(value, low, high) {
  if (high === low) return value < high ? 0 : 1;
  return clamp((value - low) / (high - low), 0, 1);
}

function dist(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function percentile(sorted, p) {
  if (!sorted.length) return 0;
  const i = (p / 100) * (sorted.length - 1);
  const lo = Math.floor(i), hi = Math.ceil(i);
  return lo === hi ? sorted[lo] : sorted[lo] + (sorted[hi] - sorted[lo]) * (i - lo);
}

export function eyeAspectRatio(pts, idx) {
  const [p1, p2, p3, p4, p5, p6] = idx.map(i => pts[i]);
  const horizontal = dist(p1, p4);
  if (horizontal <= 0) return 0;
  return (dist(p2, p6) + dist(p3, p5)) / (2 * horizontal);
}

export function mouthAspectRatio(pts) {
  const width = dist(pts[MOUTH_LEFT], pts[MOUTH_RIGHT]);
  if (width <= 0) return 0;
  return dist(pts[MOUTH_TOP], pts[MOUTH_BOTTOM]) / width;
}

/** The face's bounding box in pixels, from the landmarks. */
export function faceBox(pts) {
  if (!pts || !pts.length) return null;
  let x1 = Infinity, y1 = Infinity, x2 = -Infinity, y2 = -Infinity;
  for (const p of pts) {
    if (p.x < x1) x1 = p.x;
    if (p.y < y1) y1 = p.y;
    if (p.x > x2) x2 = p.x;
    if (p.y > y2) y2 = p.y;
  }
  return [x1, y1, x2, y2];
}

/**
 * Yaw, pitch and roll in degrees from MediaPipe's own facial
 * transformation matrix.
 *
 * The desktop version solves this with cv2.solvePnP over five
 * landmarks, which took two bugs to get right: a near-planar point set
 * has a mirror twin, so a +30 degree turn and a -30 degree turn project
 * to nearly the same picture, and the iterative solver walked between
 * them -- 31 sign crossings in 206 frames. MediaPipe returns the pose
 * directly here, which sidesteps the ambiguity rather than solving it.
 *
 * The matrix is column-major, so element (row, col) is data[col * 4 + row].
 */
export function headPose(matrix) {
  const m = matrix;
  const r = (row, col) => m[col * 4 + row];

  const pitch = Math.asin(clamp(-r(1, 2), -1, 1));
  const yaw   = Math.atan2(r(0, 2), r(2, 2));
  const roll  = Math.atan2(r(1, 0), r(1, 1));

  const deg = x => x * 180 / Math.PI;

  return { yaw: deg(yaw), pitch: deg(pitch), roll: deg(roll) };
}

// ----------------------------------------------------------------------
// EAR calibration
// ----------------------------------------------------------------------

/**
 * A fixed EAR threshold encodes an assumption about eyelid shape.
 * Eye shape varies enough between people that one driver's open eye
 * sits below another's blink threshold and is read as permanently
 * closed, which pegs PERCLOS at 100% and cries wolf for a whole drive.
 *
 * So the threshold is learned: watch this driver for a few seconds,
 * find their open and closed extremes, and place the threshold between
 * them.
 */
export class EarCalibrator {
  constructor() { this.reset(); }

  reset() {
    this.samples = [];
    this.started = null;
    this.threshold = CONFIG.EAR_THRESHOLD;
    this.calibrated = false;
    this.confidence = 0.5;
  }

  add(ear, t) {
    if (this.calibrated) return this.threshold;

    if (this.started === null) this.started = t;
    this.samples.push(ear);

    if (t - this.started < CONFIG.EAR_CALIBRATION_TIME) return this.threshold;

    if (this.samples.length >= CONFIG.EAR_MIN_SAMPLES) {
      const sorted = [...this.samples].sort((a, b) => a - b);
      const open   = percentile(sorted, CONFIG.EAR_OPEN_PERCENTILE);
      const closed = percentile(sorted, CONFIG.EAR_CLOSED_PERCENTILE);

      // Too little spread means the driver never blinked while we
      // watched, so the "closed" end is really just noise on an open
      // eye. Keep the default rather than invent a threshold from it.
      if (open - closed >= CONFIG.EAR_MIN_RANGE) {
        this.threshold = clamp(
          closed + CONFIG.EAR_CLOSED_FRACTION * (open - closed),
          CONFIG.EAR_THRESHOLD_MIN,
          CONFIG.EAR_THRESHOLD_MAX
        );
        this.confidence = 1.0;
      } else {
        this.confidence = 0.6;
      }
    }

    this.calibrated = true;
    return this.threshold;
  }

  get progress() {
    if (this.calibrated || this.started === null) return 1;
    return clamp((this.lastT - this.started) / CONFIG.EAR_CALIBRATION_TIME, 0, 1);
  }
}

// ----------------------------------------------------------------------
// blinks, PERCLOS, microsleep
// ----------------------------------------------------------------------

export class BlinkDetector {
  constructor() {
    this.calibrator = new EarCalibrator();
    this.closedSince = null;
    this.blinkCount = 0;
    this.microsleepCount = 0;
    this.blinkTimes = [];
    this.closedDuration = 0;
    this.microsleep = false;
    this.ear = null;
    this.threshold = CONFIG.EAR_THRESHOLD;
  }

  recalibrate() { this.calibrator.reset(); }

  /** Returns null when there is no face -- the same contract the Python
   *  has, and the reason the evaluation harness once counted a face gap
   *  as a burst of twenty blinks. */
  update(pts, t) {
    if (!pts) {
      this.closedSince = null;
      this.closedDuration = 0;
      this.microsleep = false;
      return null;
    }

    const ear = (eyeAspectRatio(pts, LEFT_EYE) +
                 eyeAspectRatio(pts, RIGHT_EYE)) / 2;

    this.ear = ear;
    this.calibrator.lastT = t;
    this.threshold = this.calibrator.add(ear, t);

    const closed = ear < this.threshold;

    if (closed) {
      if (this.closedSince === null) this.closedSince = t;
      this.closedDuration = t - this.closedSince;
    } else {
      if (this.closedSince !== null) {
        const held = t - this.closedSince;

        if (held >= CONFIG.MICROSLEEP_DURATION) {
          this.microsleepCount += 1;
        } else if (held >= CONFIG.BLINK_MIN_DURATION) {
          this.blinkCount += 1;
          this.blinkTimes.push(t);
        }
      }
      this.closedSince = null;
      this.closedDuration = 0;
    }

    this.microsleep = this.closedDuration >= CONFIG.MICROSLEEP_DURATION;

    while (this.blinkTimes.length &&
           t - this.blinkTimes[0] > CONFIG.BLINK_RATE_WINDOW) {
      this.blinkTimes.shift();
    }

    return {
      ear,
      threshold: this.threshold,
      eyes_closed: closed,
      closed_duration: this.closedDuration,
      microsleep: this.microsleep,
      blink_count: this.blinkCount,
      microsleep_count: this.microsleepCount,
      blink_rate: this.blinkTimes.length,
      calibrated: this.calibrator.calibrated,
      confidence: this.calibrator.confidence
    };
  }
}

export class DrowsinessDetector {
  constructor() { this.samples = []; }

  update(blink, t) {
    if (!blink) return null;

    this.samples.push([t, blink.eyes_closed]);

    while (this.samples.length &&
           t - this.samples[0][0] > CONFIG.PERCLOS_WINDOW) {
      this.samples.shift();
    }

    const span = this.samples.length ? t - this.samples[0][0] : 0;

    // Below PERCLOS_MIN_DATA seconds of history the fraction swings on
    // single frames, so no claim is made from it.
    if (span < CONFIG.PERCLOS_MIN_DATA) {
      return { perclos: 0, level: "AWAKE", ready: false, span };
    }

    const closed = this.samples.filter(s => s[1]).length;
    const perclos = closed / this.samples.length;

    let level = "AWAKE";
    if (perclos >= CONFIG.PERCLOS_DROWSY) level = "DROWSY";
    else if (perclos >= CONFIG.PERCLOS_TIRED) level = "TIRED";

    return { perclos, level, ready: true, span };
  }
}

// ----------------------------------------------------------------------
// yawns
// ----------------------------------------------------------------------

export class YawnDetector {
  constructor() {
    this.openSince = null;
    this.yawnCount = 0;
    this.times = [];
    this.mar = 0;
  }

  update(pts, t) {
    if (!pts) { this.openSince = null; return null; }

    const mar = mouthAspectRatio(pts);
    this.mar = mar;

    const open = mar > CONFIG.MAR_THRESHOLD;

    if (open) {
      if (this.openSince === null) this.openSince = t;
    } else {
      // The yawn is counted when the mouth closes again, which is the
      // instant the detector can be sure how long it lasted.
      if (this.openSince !== null &&
          t - this.openSince >= CONFIG.YAWN_MIN_DURATION) {
        this.yawnCount += 1;
        this.times.push(t);
      }
      this.openSince = null;
    }

    while (this.times.length && t - this.times[0] > CONFIG.YAWN_RATE_WINDOW) {
      this.times.shift();
    }

    return {
      mar,
      yawning: open && this.openSince !== null &&
               t - this.openSince >= CONFIG.YAWN_MIN_DURATION,
      yawn_count: this.yawnCount,
      recent_yawns: this.times.length
    };
  }
}

// ----------------------------------------------------------------------
// head pose
// ----------------------------------------------------------------------

export class HeadPoseEstimator {
  constructor() { this.startCalibration(); }

  startCalibration() {
    this.calStarted = null;
    this.calYaw = [];
    this.calPitch = [];
    this.offsetYaw = 0;
    this.offsetPitch = 0;
    this.calibrated = false;
  }

  update(matrix, mar, t) {
    if (!matrix) return null;

    const raw = headPose(matrix);

    // A pose the estimator will not vouch for is not evidence. solvePnP
    // fits a rigid head and a wide open mouth is a face that has
    // stopped being one; on a desktop recording a yawn with the driver
    // facing straight ahead produced 28 degrees of reported yaw and a
    // logged distraction. MediaPipe's own solve is steadier, but the
    // mouth still deforms the landmarks it fits, so the gate stays.
    const reliable = mar === null || mar <= CONFIG.HEAD_POSE_MAX_MAR;

    if (!this.calibrated && reliable) {
      if (this.calStarted === null) this.calStarted = t;

      this.calYaw.push(raw.yaw);
      this.calPitch.push(raw.pitch);

      if (t - this.calStarted >= CONFIG.HEAD_POSE_CALIBRATION_TIME) {
        const mean = a => a.reduce((s, v) => s + v, 0) / a.length;

        // Clamped, because a driver who looked away during calibration
        // must not teach the system that sideways is straight ahead.
        this.offsetYaw = clamp(mean(this.calYaw),
          -CONFIG.HEAD_POSE_MAX_OFFSET, CONFIG.HEAD_POSE_MAX_OFFSET);
        this.offsetPitch = clamp(mean(this.calPitch),
          -CONFIG.HEAD_POSE_MAX_OFFSET, CONFIG.HEAD_POSE_MAX_OFFSET);
        this.calibrated = true;
      }
    }

    return {
      yaw: raw.yaw - this.offsetYaw,
      pitch: raw.pitch - this.offsetPitch,
      roll: raw.roll,
      calibrated: this.calibrated,
      reliable,
      progress: this.calStarted === null ? 0 :
        clamp((t - this.calStarted) / CONFIG.HEAD_POSE_CALIBRATION_TIME, 0, 1)
    };
  }
}

// ----------------------------------------------------------------------
// distraction
// ----------------------------------------------------------------------

export class DistractionDetector {
  constructor() {
    this.offRoadSince = null;
    this.onRoadSince = null;
    this.lastSeen = null;
    this.active = false;
    this.count = 0;
  }

  update(pose, t) {
    // No reading at all, or one the estimator will not vouch for.
    //
    // A head turned far enough to be a distraction is a head the
    // landmarker often cannot find, so the gaps land inside the very
    // episodes this detector exists to catch. The clock is held across
    // a short gap -- but nothing is claimed while it lasts, because
    // holding the clock is not the same as claiming to see.
    if (!pose || !pose.reliable) {
      this.active = false;

      if (this.lastSeen !== null &&
          t - this.lastSeen > CONFIG.DISTRACTION_FACE_GAP) {
        this.offRoadSince = null;
        this.onRoadSince = null;
      }
      return null;
    }

    this.lastSeen = t;

    if (!pose.calibrated) return null;

    const offRoad = Math.abs(pose.yaw) > CONFIG.OFF_ROAD_YAW ||
                    Math.abs(pose.pitch) > CONFIG.OFF_ROAD_PITCH;

    if (offRoad) {
      this.onRoadSince = null;
      if (this.offRoadSince === null) this.offRoadSince = t;
    } else if (this.offRoadSince !== null) {
      // Ending an episode needs evidence too. A head cannot swing fifty
      // degrees and back inside a tenth of a second, so a frame that
      // says it did is a bad fit: on a recorded nine second turn one
      // reading of 9 degrees sat between two of 54, and everything
      // after it was thrown away.
      if (this.onRoadSince === null) this.onRoadSince = t;

      if (t - this.onRoadSince >= CONFIG.DISTRACTION_RECOVERY) {
        this.offRoadSince = null;
        this.onRoadSince = null;
      }
    }

    const duration = this.offRoadSince === null ? 0 : t - this.offRoadSince;
    const distracted = duration >= CONFIG.DISTRACTION_DURATION;

    if (distracted && !this.active) this.count += 1;
    this.active = distracted;

    return {
      eyes_on_road: !offRoad,
      distracted,
      off_road_duration: duration,
      distraction_count: this.count
    };
  }
}

// ----------------------------------------------------------------------
// the camera itself
// ----------------------------------------------------------------------

/**
 * Whether the monitor can see at all.
 *
 * Two ways for it not to be able to, and they are not the same:
 *
 *   blocked - a flat, dark picture. Told apart from a dark cabin by
 *             variance, because a dark cabin still has shapes,
 *             highlights and sensor noise in it and a covered lens
 *             does not.
 *
 *   frozen  - the picture has stopped changing. Two consecutive frames
 *             from a live sensor are never bit identical, so a run of
 *             identical ones is not a dark room, it is a still image
 *             being fed in where a camera should be. In a fleet this
 *             is how the system gets defeated, and until the desktop
 *             evaluation went looking for it, it was not checked at
 *             all: a frozen feed simply read as a calm, safe drive.
 */
export class CameraMonitor {
  constructor() {
    this.darkSince = null;
    this.identicalSince = null;
    this.fingerprint = null;
    this.blocked = false;
    this.frozen = false;
  }

  /** `fingerprint` is any cheap digest of the frame's pixels; equal
   *  fingerprints on consecutive frames is what "identical" means. */
  update(brightness, variance, fingerprint, t) {
    const flat = brightness < CONFIG.CAMERA_BLOCKED_BRIGHTNESS &&
                 variance < CONFIG.CAMERA_BLOCKED_VARIANCE;

    if (flat) {
      if (this.darkSince === null) this.darkSince = t;
    } else {
      this.darkSince = null;
    }

    const darkFor = this.darkSince === null ? 0 : t - this.darkSince;
    const frozenFor = this._frozenFor(fingerprint, t);

    this.frozen = frozenFor >= CONFIG.CAMERA_FROZEN_TIME;

    // Both must hold for a while, so the first frames of a camera that
    // has not woken up yet do not raise it.
    const dark = darkFor >= CONFIG.CAMERA_BLOCKED_TIME;
    this.blocked = dark || this.frozen;

    return {
      blocked: this.blocked,
      frozen: this.frozen,
      brightness,
      variance,
      duration: this.blocked ? Math.max(darkFor, frozenFor) : 0,
      reason: this.frozen ? "frozen" : dark ? "flat" : null
    };
  }

  _frozenFor(fingerprint, t) {
    if (fingerprint === null || fingerprint === undefined) {
      this.fingerprint = null;
      this.identicalSince = null;
      return 0;
    }

    if (fingerprint !== this.fingerprint) {
      this.fingerprint = fingerprint;
      this.identicalSince = t;
      return 0;
    }

    return t - this.identicalSince;
  }
}

// ----------------------------------------------------------------------
// hands on the wheel
// ----------------------------------------------------------------------

/**
 * Whether the driver's hands are on the wheel -- and, first, whether
 * that question can be answered at all.
 *
 * This camera is aimed at a face, and on most setups the wheel is below
 * the bottom edge, so "no hands found" is the normal state rather than
 * a warning. The monitor therefore has to earn the right to complain,
 * and what earns it is hands being found in a decent SHARE of the
 * recent past rather than a NUMBER of times.
 *
 * The count was the first attempt and it was far too cheap. Hands are
 * looked for on one frame in six, so ten sightings is two seconds of
 * one continuous sighting, and two seconds bought sixty seconds of
 * judgement. Measured on a recording: the driver held a phone up beside
 * their face, the monitor decided after 1.7 s of that it could see the
 * wheel, and it then called the next thirty seconds -- hands resting in
 * a lap, below the frame -- hands off the wheel. 1520 frames reported
 * against 349 performed, 22.9% precision.
 *
 * A share is the right shape for the question. A wheel that is in shot
 * has hands on it nearly all the time, so the real figure is close to
 * 100% and even a long legitimate spell off the wheel leaves it high. A
 * wheel that is out of shot produces hands only in passing -- a
 * gesture, a scratch, a phone -- which is a small fraction of a minute
 * however vividly it happens.
 */
export class WheelMonitor {
  constructor() {
    this.samples = [];       // [timestamp, found]
    this.sightings = 0;
    this.proved = false;
    this.missingSince = null;
    this.handsOff = false;
    this.episodes = 0;
  }

  get canSee() { return this.proved; }

  /**
   * `hands` is a list of hand landmark arrays in normalized
   * coordinates, or null when the detector did not run this frame.
   * `wheelLine` is where the wheel is assumed to begin, as a fraction
   * of frame height: hands above it are up near the face and are not
   * on a wheel.
   */
  update(hands, wheelLine, t) {
    if (hands !== null) {
      const found = this._atWheel(hands, wheelLine);
      this.samples.push([t, found]);
      if (found) this.sightings += 1;
      this._reviewEvidence(t);

      if (found) {
        this.missingSince = null;
      } else if (this.missingSince === null) {
        this.missingSince = t;
      }
    }

    const missingFor = this.missingSince === null ? 0 : t - this.missingSince;

    // The alarm needs both: evidence that the wheel is visible at all,
    // and HANDS_OFF_TIME of it being empty.
    const off = this.proved && missingFor >= CONFIG.HANDS_OFF_TIME;

    if (off && !this.handsOff) this.episodes += 1;
    this.handsOff = off;

    return {
      can_see: this.proved,
      hands_off: off,
      missing_for: missingFor,
      share: this.samples.length ? this.sightings / this.samples.length : 0,
      hand_count: hands === null ? null : this._count(hands, wheelLine),
      episodes: this.episodes
    };
  }

  /** Only hands low enough in frame to be at a wheel. A hand raised
   *  beside the face is the gesture this used to mistake for proof. */
  _atWheel(hands, wheelLine) {
    return this._count(hands, wheelLine) > 0;
  }

  _count(hands, wheelLine) {
    return hands.filter(hand => {
      const mean = hand.reduce((s, p) => s + p.y, 0) / hand.length;
      return mean >= wheelLine;
    }).length;
  }

  _reviewEvidence(t) {
    while (this.samples.length &&
           t - this.samples[0][0] > CONFIG.HANDS_SEEN_WINDOW) {
      const [, found] = this.samples.shift();
      if (found) this.sightings -= 1;
    }

    if (!this.samples.length) return;

    // The window has to have had time to fill before the share means
    // anything, or the first hand of the drive is 100% of the evidence.
    const span = this.samples[this.samples.length - 1][0] - this.samples[0][0];
    if (span < CONFIG.HANDS_SEEN_WARMUP) return;

    if (this.sightings / this.samples.length >= CONFIG.HANDS_SEEN_SHARE) {
      this.proved = true;
    } else if (this.sightings === 0) {
      // Only a whole window with nothing in it withdraws the claim. A
      // long legitimate spell off the wheel drives the share down, and
      // it must not take the ability to notice down with it.
      this.proved = false;
    }
  }
}

// ----------------------------------------------------------------------
// phone
// ----------------------------------------------------------------------

/**
 * A phone in the frame, pooled over time.
 *
 * A phone must be seen for PHONE_MIN_DURATION before it counts as in
 * use, and the usage ends once none has been seen for
 * PHONE_RELEASE_TIME. That rides out single-frame false detections in
 * one direction and missed frames in the other.
 */
export class PhoneMonitor {
  constructor() {
    this.seenSince = null;
    this.lastSeen = null;
    this.inUse = false;
    this.episodes = 0;
    this.phones = [];
  }

  /** `detections` is a list of {confidence}, or null when the detector
   *  did not run this frame. */
  update(detections, t) {
    if (detections !== null) {
      const phones = detections.filter(d => d.confidence >= CONFIG.PHONE_CONFIDENCE);
      this.phones = phones;

      if (phones.length) {
        if (this.seenSince === null) this.seenSince = t;
        this.lastSeen = t;
      } else if (this.lastSeen !== null &&
                 t - this.lastSeen >= CONFIG.PHONE_RELEASE_TIME) {
        this.seenSince = null;
      }
    }

    const held = this.seenSince === null ? 0 : t - this.seenSince;
    const using = held >= CONFIG.PHONE_MIN_DURATION;

    if (using && !this.inUse) this.episodes += 1;
    this.inUse = using;

    return {
      phone_in_use: using,
      usage_duration: held,
      phones: this.phones,
      episodes: this.episodes
    };
  }
}

// ----------------------------------------------------------------------
// seatbelt
// ----------------------------------------------------------------------

// Chest region, relative to the face box: from ROI_TOP to ROI_BOTTOM
// face heights below the chin, and ROI_SIDE face widths either side.
const ROI_SIDE = 0.8, ROI_TOP = 0.1, ROI_BOTTOM = 1.6;

// Less chest than this in view (in face heights) and no judgement is
// made: the strap would not be seen reaching the centre line.
const MIN_VISIBLE_HEIGHT = 0.9;

const MIN_EDGE_LENGTH = 0.35;      // of the region's height
const PARALLEL_TOLERANCE = 8.0;    // degrees
const MIN_OVERLAP = 0.5;           // of the shorter edge
const MIDLINE_SHARE = 0.15;

// The region is scaled to this width before the edge work. Every test
// in the chain is a RATIO -- strap width against face width, edge
// length against region height -- so scaling preserves all of them,
// and a Hough transform over a 640-wide strip on a phone would not
// finish inside a frame.
const WORK_WIDTH = 200;

/**
 * Classical computer-vision baseline for seatbelt detection.
 *
 * No pretrained seatbelt model ships with the common libraries, and
 * training one needs a labelled in-cabin dataset. Until then, this
 * looks for the strap's visual signature in the chest region below the
 * driver's face: two long, parallel, diagonal edges (the strap's two
 * sides) a strap-width apart, crossing the body's centre line. Both
 * diagonals are accepted, so left- and right-hand-drive cars both work.
 *
 * On a phone on a windscreen mount the chest is usually out of frame,
 * and the honest answer to "is the belt on" is then "I cannot see".
 * That is what `visible: false` means, and the pooling below turns it
 * into UNKNOWN rather than into an accusation.
 */
export function detectSeatbelt(ctx, box, frameWidth, frameHeight) {
  const blank = { visible: false, belt_found: false, roi: null, strap: null };

  if (!box) return blank;

  const [fx1, fy1, fx2, fy2] = box;
  const faceW = fx2 - fx1, faceH = fy2 - fy1;
  if (faceW <= 0 || faceH <= 0) return blank;

  const x1 = Math.round(Math.max(fx1 - ROI_SIDE * faceW, 0));
  const x2 = Math.round(Math.min(fx2 + ROI_SIDE * faceW, frameWidth));
  const y1 = Math.round(Math.min(fy2 + ROI_TOP * faceH, frameHeight));
  const y2 = Math.round(Math.min(fy2 + ROI_BOTTOM * faceH, frameHeight));

  if (y2 - y1 < MIN_VISIBLE_HEIGHT * faceH) return blank;

  // The strap needs room to be seen crossing the chest.
  if (x2 - x1 < 2 * faceW) return blank;

  const roi = [x1, y1, x2, y2];
  const width = x2 - x1, height = y2 - y1;

  const scale = Math.min(1, WORK_WIDTH / width);
  const w = Math.max(8, Math.round(width * scale));
  const h = Math.max(8, Math.round(height * scale));

  let pixels;
  try {
    pixels = ctx.getImageData(x1, y1, width, height);
  } catch (e) {
    return blank;
  }

  const small = downscale(pixels, width, height, w, h);

  let gray = grayscale(small, w, h);
  gray = equalize(gray);
  gray = blur(gray, w, h);

  // Canny thresholds that follow the region's brightness, as the
  // Python's do -- but off a clamped median, for the reason given
  // where medianForThresholds is defined.
  const mid = medianForThresholds(gray);
  const edges = canny(gray, w, h, Math.max(0, 0.66 * mid), Math.min(255, 1.33 * mid));

  const found = segments(edges, w, h, {
    minAngle: CONFIG.SEATBELT_MIN_ANGLE,
    maxAngle: CONFIG.SEATBELT_MAX_ANGLE,
    minLength: MIN_EDGE_LENGTH * h,
    maxGap: Math.round(0.05 * h) + 1
  });

  // Centre line of the body, taken from the face, in region
  // coordinates and at the working scale.
  const midline = ((fx1 + fx2) / 2 - x1) * scale;

  const strap = findStrap(found, faceW * scale, midline, {
    minWidth: CONFIG.SEATBELT_MIN_WIDTH,
    maxWidth: CONFIG.SEATBELT_MAX_WIDTH,
    parallelTolerance: PARALLEL_TOLERANCE,
    minOverlap: MIN_OVERLAP,
    midlineShare: MIDLINE_SHARE
  });

  return {
    visible: true,
    belt_found: strap !== null,
    roi,
    strap: strap === null ? null : strap.map(e => [
      x1 + e.x1 / scale, y1 + e.y1 / scale,
      x1 + e.x2 / scale, y1 + e.y2 / scale
    ])
  };
}

/** Nearest-neighbour, which is all a gradient operator needs and a
 *  great deal cheaper than drawing through another canvas. */
function downscale(pixels, width, height, w, h) {
  const src = pixels.data;
  const out = new Uint8ClampedArray(w * h * 4);

  for (let y = 0; y < h; y++) {
    const sy = Math.min(height - 1, Math.floor(y * height / h));
    for (let x = 0; x < w; x++) {
      const sx = Math.min(width - 1, Math.floor(x * width / w));
      const from = (sy * width + sx) * 4, to = (y * w + x) * 4;
      out[to] = src[from];
      out[to + 1] = src[from + 1];
      out[to + 2] = src[from + 2];
      out[to + 3] = 255;
    }
  }

  return out;
}

/**
 * Pools per-frame seatbelt observations over a time window into a
 * steady status:
 *
 *  - ON:      found in at least ON_RATIO of the recent frames where the
 *             chest was visible.
 *  - OFF:     found in OFF_RATIO or fewer of them.
 *  - UNKNOWN: the chest was not visible in most recent frames.
 *
 * Between the two ratios the previous status holds, so a borderline
 * scene does not flicker.
 */
export class SeatbeltMonitor {
  // Fewer visible frames than this in the window and no judgement is
  // made.
  static MIN_VISIBLE_SAMPLES = 3;

  constructor() {
    this.samples = [];      // [timestamp, visible, found]
    this.status = "UNKNOWN";
    this.lastObservation = null;
  }

  update(observation, t) {
    if (observation) {
      this.samples.push([t, observation.visible, observation.belt_found]);
      this.lastObservation = observation;
    }

    while (this.samples.length &&
           t - this.samples[0][0] > CONFIG.SEATBELT_WINDOW) {
      this.samples.shift();
    }

    const visible = this.samples.filter(s => s[1]).map(s => s[2]);
    let ratio = 0;

    if (visible.length < SeatbeltMonitor.MIN_VISIBLE_SAMPLES ||
        visible.length < this.samples.length / 2) {
      this.status = "UNKNOWN";
    } else {
      ratio = visible.filter(Boolean).length / visible.length;

      if (ratio >= CONFIG.SEATBELT_ON_RATIO) this.status = "ON";
      else if (ratio <= CONFIG.SEATBELT_OFF_RATIO) this.status = "OFF";
      else if (this.status === "UNKNOWN") {
        // No previous status to hold; lean to the safe side.
        this.status = "OFF";
      }
    }

    return { status: this.status, ratio, observation: this.lastObservation };
  }
}

// ----------------------------------------------------------------------
// expression
// ----------------------------------------------------------------------

// Blendshapes that together read as a negative, driving-relevant
// expression, with the weight each carries. Brow and mouth lead,
// because they are what separates anger and distress from a neutral
// face; eye squint and jaw tension support rather than decide.
const NEGATIVE_SHAPES = {
  browDownLeft: 1.0, browDownRight: 1.0,
  mouthFrownLeft: 1.0, mouthFrownRight: 1.0,
  browInnerUp: 0.6,
  eyeSquintLeft: 0.4, eyeSquintRight: 0.4,
  mouthPressLeft: 0.3, mouthPressRight: 0.3,
  jawForward: 0.3
};

/**
 * NOT a port of the FER+ emotion classifier. This is a substitute, and
 * the difference matters enough to say in the file.
 *
 * The desktop runs emotion-ferplus-8.onnx: a convolutional network over
 * a cropped grey face, eight labelled classes, the most expensive model
 * in that pipeline (~67 ms a frame against ~16 ms for the landmarks)
 * carrying the smallest penalty in the budget. Shipping it here would
 * mean an ONNX runtime plus a ~35 MB download, on a phone, for 10
 * points.
 *
 * MediaPipe's face landmarker already produces 52 ARKit blendshape
 * scores from the same forward pass that produces the landmarks, at no
 * additional cost. A weighted sum of the ones that make up a negative
 * expression is a cruder instrument than a trained classifier -- it
 * names no emotion, and it would not survive comparison on a labelled
 * benchmark -- but it measures the same thing closely enough to drive
 * the same 10-point penalty, and the smoothing and alert threshold are
 * the FER+ ones unchanged.
 *
 * So the condition is called `negative_emotion` for continuity with the
 * desktop's scoring, and the banner says NEGATIVE EXPRESSION, because
 * that is what is actually being measured.
 */
export class ExpressionAnalyzer {
  constructor() {
    this.score = 0;
    this.available = false;
  }

  /** `blendshapes` is MediaPipe's categories array, or null. */
  update(blendshapes) {
    if (!blendshapes || !blendshapes.length) {
      return this.available
        ? { confidence: this.score, negative: false, available: true }
        : null;
    }

    this.available = true;

    let total = 0, weight = 0;
    for (const category of blendshapes) {
      const w = NEGATIVE_SHAPES[category.categoryName];
      if (w === undefined) continue;
      total += category.score * w;
      weight += w;
    }

    const raw = weight > 0 ? total / weight : 0;

    // A single expressive frame is not a mood. Same smoothing weight
    // the FER+ analyser uses.
    this.score += (raw - this.score) * CONFIG.EMOTION_SMOOTHING;

    return {
      confidence: this.score,
      negative: this.score >= CONFIG.EMOTION_ALERT_CONFIDENCE,
      available: true
    };
  }
}

// ----------------------------------------------------------------------
// driving context
// ----------------------------------------------------------------------

// Kept here rather than imported from weather.js so the scoring layer
// has no dependency on anything that touches the network.
const WEATHER_WEIGHTS = {
  clear: 1.00, cloudy: 1.00, rain: 1.15, heavy_rain: 1.25,
  fog: 1.25, snow: 1.30, ice: 1.35
};

/**
 * The part of the risk picture that is not the driver.
 *
 * The same microsleep is not equally dangerous at every moment. Time of
 * day, time already spent at the wheel, how much light the camera has
 * and what the road is like all change how much weight a detection
 * deserves.
 *
 * Only multipliers of 1.0 or more are produced: context amplifies risk
 * and never discounts it, so the score can never be more permissive
 * than the flat baseline.
 */
export class DrivingContext {
  constructor({ hour = 12, elapsed = 0, brightness = null,
                weather = "clear", weatherSource = "default" } = {}) {
    this.hour = hour;
    this.elapsed = elapsed;
    this.brightness = brightness;
    this.weather = weather;
    this.weatherSource = weatherSource;
  }

  /** 0 in a well-lit cabin, 1 in a dark one. */
  get darkness() {
    if (this.brightness === null) return 0;
    return 1 - ramp(this.brightness, CONFIG.BRIGHTNESS_DARK, CONFIG.BRIGHTNESS_LIT);
  }

  circadianMultiplier() {
    return circadianBand(this.hour).multiplier;
  }

  timeOnTaskMultiplier() {
    const hours = this.elapsed / 3600;
    return 1 + CONFIG.TIME_ON_TASK_MAX * ramp(hours, 0, CONFIG.TIME_ON_TASK_HOURS);
  }

  /**
   * How much the conditions outside raise the cost of a lapse.
   *
   * Unlike the circadian and time-on-task multipliers, this says
   * nothing about how LIKELY a lapse is -- rain does not make a driver
   * sleepier. It is about consequence: there is less room to recover on
   * a wet road, so the same lapse is worth more. That argument holds
   * for a microsleep and a glance away alike, so this amplifies both
   * families.
   *
   * Anything unrecognised weighs 1.0, so a string nobody expected
   * cannot quietly inflate the score.
   */
  weatherMultiplier() {
    return WEATHER_WEIGHTS[this.weather] ?? 1.0;
  }

  fatigueMultiplier() {
    return this.circadianMultiplier() *
           this.timeOnTaskMultiplier() *
           this.weatherMultiplier();
  }

  /** Less visibility leaves less margin for a glance away, so the
   *  attention conditions weigh more in the dark and in bad weather. */
  attentionMultiplier() {
    return (1 + CONFIG.LOW_LIGHT_ATTENTION_MAX * this.darkness) *
           this.weatherMultiplier();
  }

  summary() {
    const parts = [];
    if (this.circadianMultiplier() > 1.0) {
      parts.push(circadianBand(this.hour).label);
    }
    if (this.timeOnTaskMultiplier() > 1.01) {
      parts.push(`${(this.elapsed / 3600).toFixed(1)}h`);
    }
    if (this.weatherMultiplier() > 1.0) {
      parts.push(this.weather.replace(/_/g, " "));
    }
    if (this.darkness > 0.05) parts.push("dark");
    return parts.length ? parts.join(" ") : null;
  }
}

// ----------------------------------------------------------------------
// the three weighting layers
// ----------------------------------------------------------------------

function scaled(value, span, floor) {
  return floor + (1 - floor) * ramp(value, span[0], span[1]);
}

/**
 * How bad THIS instance is: a six second glance away is worse than a
 * two second one, and the flat table charged the same for both.
 *
 * Can only ever reduce a penalty, so `base` stays the documented
 * maximum any single condition can cost and the 100-point budget keeps
 * its meaning.
 */
export function severity(condition, state, absentDuration) {
  const span = SEVERITY_SPANS[condition];
  if (!span) return 1.0;      // no_seatbelt and negative_emotion are
                              // states, not episodes: there is no
                              // duration that makes them worse.

  let value;
  if (condition === "microsleep") value = state.blink?.closed_duration;
  else if (condition === "drowsy" || condition === "tired") value = state.drowsiness?.perclos;
  else if (condition === "distracted") value = state.distraction?.off_road_duration;
  else if (condition === "phone") value = state.phone?.usage_duration;
  else if (condition === "hands_off") value = state.wheel?.missing_for;
  else if (condition === "frequent_yawning") value = state.yawn?.recent_yawns;
  else if (condition === "camera_blocked") value = state.camera?.duration;
  else if (condition === "driver_absent") value = absentDuration;

  if (value === null || value === undefined) return 1.0;

  return scaled(value, span, CONFIG.SEVERITY_FLOOR);
}

/**
 * How much the detector behind `condition` deserves to be trusted this
 * frame. Never zero: a condition that is active still shows its alert
 * at full strength, and only its contribution to the score is scaled.
 */
export function confidence(condition, state, darkness = 0) {
  const floor = v => clamp(v, CONFIG.CONFIDENCE_FLOOR, 1);

  if (EYE_CONDITIONS.has(condition)) {
    return floor(eyeConfidence(state, darkness));
  }

  if (condition === "no_seatbelt") return floor(seatbeltConfidence(state));

  if (condition === "negative_emotion") {
    const value = state.emotion?.confidence;
    if (value === null || value === undefined) return 1.0;
    return floor(scaled(value, CONFIDENCE_EMOTION, CONFIDENCE_EMOTION[2]));
  }

  if (condition === "phone") return floor(phoneConfidence(state));

  return 1.0;
}

/**
 * Eye measurements depend on readable eyelids. Two things erode them: a
 * calibration the EAR calibrator could not complete well, and a dark
 * cabin, where the landmarks get noisy.
 *
 * This is the counterweight to the circadian multiplier. Night raises
 * the risk a drowsy driver poses, but it lowers the camera's ability to
 * see that driver, and without this the engine would amplify its own
 * noise at exactly the hour false alarms do the most damage.
 */
function eyeConfidence(state, darkness) {
  const calibration = state.blink?.confidence;
  const weight = calibration === null || calibration === undefined ? 1 : calibration;

  // A face at a steep angle is a face whose eyelids are foreshortened.
  const turned = Math.min(Math.abs(state.head_pose?.yaw ?? 0) / 60, 1);

  return weight *
         (1 - (1 - CONFIG.CONFIDENCE_EYES_DARK) * darkness) *
         (1 - 0.4 * turned);
}

/**
 * The pooled ratio is how many of the recent visible frames contained a
 * strap. A verdict of OFF is most doubtful just under the boundary and
 * most certain when nothing strap-shaped was seen at all.
 */
function seatbeltConfidence(state) {
  const ratio = state.seatbelt?.ratio;
  if (ratio === null || ratio === undefined) return 1.0;

  const certainty = 1 - ramp(ratio, 0, CONFIG.SEATBELT_OFF_RATIO);

  return CONFIG.CONFIDENCE_SEATBELT_AT_BOUNDARY +
         (1 - CONFIG.CONFIDENCE_SEATBELT_AT_BOUNDARY) * certainty;
}

function phoneConfidence(state) {
  const phones = state.phone?.phones;
  if (!phones || !phones.length) return 1.0;
  const best = Math.max(...phones.map(p => p.confidence ?? 1));
  return scaled(best, CONFIDENCE_PHONE, CONFIDENCE_PHONE[2]);
}

// ----------------------------------------------------------------------
// the safety index
// ----------------------------------------------------------------------

export class DriverSafetyIndex {
  constructor() {
    this.score = null;
    this.level = "SAFE";
    this.last = null;
    this.active = new Set();
    this.faceLostSince = null;
    this.contributions = [];
  }

  /**
   * `context` is a DrivingContext. A bare number is accepted as the
   * hour alone, which is what the scoring tests use.
   */
  update(state, t, context) {
    const ctx = context instanceof DrivingContext
      ? context
      : new DrivingContext({ hour: typeof context === "number" ? context : 12 });

    const active = new Set();

    if (!state.face_detected) {
      if (this.faceLostSince === null) this.faceLostSince = t;
    } else {
      this.faceLostSince = null;
    }

    const absentDuration = this.faceLostSince === null
      ? null : t - this.faceLostSince;

    if (absentDuration !== null && absentDuration >= CONFIG.DRIVER_ABSENT_TIME) {
      active.add("driver_absent");
    }

    if (state.blink?.microsleep) active.add("microsleep");
    if (state.drowsiness?.level === "DROWSY") active.add("drowsy");
    if (state.drowsiness?.level === "TIRED") active.add("tired");
    if (state.distraction?.distracted) active.add("distracted");
    if ((state.yawn?.recent_yawns ?? 0) >= CONFIG.YAWN_ALERT_COUNT) {
      active.add("frequent_yawning");
    }
    if (state.camera?.blocked) active.add("camera_blocked");
    if (state.phone?.phone_in_use) active.add("phone");
    if (state.wheel?.hands_off) active.add("hands_off");
    if (state.seatbelt?.status === "OFF") active.add("no_seatbelt");
    if (state.emotion?.negative) active.add("negative_emotion");

    // A microsleep is already the worst thing the fatigue family can
    // say, so saying it twice would charge one cause two penalties.
    if (active.has("microsleep")) { active.delete("drowsy"); active.delete("tired"); }
    if (active.has("drowsy")) active.delete("tired");

    // A blinded camera is not also an absent driver. The face is
    // missing either way, and charging both would bill one fault twice
    // -- and the louder of the two is the one the driver can act on.
    if (active.has("camera_blocked")) active.delete("driver_absent");

    const darkness = ctx.darkness;

    const contributions = [];
    let total = 0;

    for (const condition of active) {
      const base = RISK_PENALTIES[condition] ?? 0;
      const sev = severity(condition, state, absentDuration);
      const conf = confidence(condition, state, darkness);
      const mult = FATIGUE.has(condition) ? ctx.fatigueMultiplier()
                 : ATTENTION.has(condition) ? ctx.attentionMultiplier()
                 : 1.0;

      const penalty = base * sev * conf * mult;
      total += penalty;
      contributions.push({ condition, penalty, sev, conf, ctx: mult });
    }

    contributions.sort((a, b) => b.penalty - a.penalty);
    this.contributions = contributions;

    const raw = Math.max(0, 100 - total);

    // The displayed score falls quickly when a risk appears and
    // recovers slowly once it is gone: a driver should see the cost
    // immediately and earn the recovery back.
    if (this.score === null || this.last === null) {
      this.score = raw;
    } else {
      const dt = Math.max(t - this.last, 0);
      const tau = raw < this.score ? CONFIG.SCORE_FALL_TIME : CONFIG.SCORE_RISE_TIME;
      const alpha = tau <= 0 ? 1 : 1 - Math.exp(-dt / tau);
      this.score += (raw - this.score) * alpha;
    }

    this.last = t;

    this.level = this.score >= CONFIG.SAFE_SCORE ? "SAFE"
               : this.score >= CONFIG.CAUTION_SCORE ? "CAUTION" : "DANGER";

    const events = [];
    for (const [name, message, sev] of CONDITIONS) {
      if (active.has(name) && !this.active.has(name)) {
        events.push({ type: name, message, severity: sev });
      }
    }
    this.active = active;

    let alert = null;
    for (const [name, message, sev] of CONDITIONS) {
      if (active.has(name)) {
        alert = { name, message, spoken: SPOKEN[name], severity: sev };
        break;
      }
    }

    return {
      score: this.score,
      level: this.level,
      active,
      alert,
      events,
      contributions,
      context: ctx,
      absent_duration: absentDuration
    };
  }
}

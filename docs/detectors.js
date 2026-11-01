/*
 * The analytics layer, ported from analytics/ and risk_engine/.
 *
 * Every threshold here is the value from app/config.py, named the same
 * way, so the two implementations can be compared line by line. Where
 * the Python carries a comment explaining why a number is what it is,
 * that comment is carried over too -- a constant without its reason is
 * a constant nobody will dare change.
 *
 * What is deliberately NOT here: the YOLO face detector (redundant when
 * the phone is a mount's distance from the face), FER+ emotion (the
 * most expensive model in the pipeline for the smallest penalty in the
 * budget), the seatbelt detector (needs chest in view) and phone
 * detection (needs a second model). The fatigue and distraction core is
 * what this carries, and it is the part the desktop evaluation actually
 * measured.
 */

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

  // --- scoring ---
  SAFE_SCORE: 75,
  CAUTION_SCORE: 50,
  SCORE_FALL_TIME: 0.5,
  SCORE_RISE_TIME: 5.0,
  SEVERITY_FLOOR: 0.4
};

export const RISK_PENALTIES = {
  microsleep: 60,
  distracted: 30,
  driver_absent: 30,
  drowsy: 30,
  tired: 15,
  frequent_yawning: 10
};

// (condition, message, severity) most urgent first. The first active
// condition is the one shown in the banner.
export const CONDITIONS = [
  ["microsleep",       "MICROSLEEP - WAKE UP",  "critical"],
  ["distracted",       "EYES OFF ROAD",         "high"],
  ["driver_absent",    "DRIVER NOT VISIBLE",    "high"],
  ["drowsy",           "DROWSINESS DETECTED",   "high"],
  ["tired",            "SIGNS OF FATIGUE",      "medium"],
  ["frequent_yawning", "FREQUENT YAWNING",      "medium"]
];

const SEVERITY_SPANS = {
  microsleep:       [CONFIG.MICROSLEEP_DURATION, 4.0],
  drowsy:           [CONFIG.PERCLOS_DROWSY, 0.45],
  tired:            [CONFIG.PERCLOS_TIRED, CONFIG.PERCLOS_DROWSY],
  distracted:       [CONFIG.DISTRACTION_DURATION, 6.0],
  frequent_yawning: [CONFIG.YAWN_ALERT_COUNT, 8],
  driver_absent:    [CONFIG.DRIVER_ABSENT_TIME, 8.0]
};

// Hours when fatigue lapses are most likely to be real, applied to the
// fatigue conditions only. Follows published work on when fatigue
// crashes happen; see the README.
const CIRCADIAN = [
  [2, 6, 1.40],
  [0, 2, 1.20],
  [6, 7, 1.20],
  [14, 16, 1.15]
];

const FATIGUE = new Set(["microsleep", "drowsy", "tired", "frequent_yawning"]);

// MediaPipe Face Mesh indices, the same ones the Python uses.
const LEFT_EYE  = [33, 160, 158, 133, 153, 144];
const RIGHT_EYE = [362, 385, 387, 263, 373, 380];
const MOUTH_LEFT = 61, MOUTH_RIGHT = 291, MOUTH_TOP = 13, MOUTH_BOTTOM = 14;

// ----------------------------------------------------------------------
// helpers
// ----------------------------------------------------------------------

const clamp = (v, lo, hi) => Math.min(Math.max(v, lo), hi);

function ramp(value, low, high) {
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

/** Pixel coordinates, because an aspect ratio of 4:3 makes normalized
 *  distances lie: a vertical eye opening and a horizontal one are not
 *  on the same scale until both are in pixels. */
function points(landmarks, width, height) {
  return landmarks.map(p => ({ x: p.x * width, y: p.y * height }));
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

    const span = this.samples.length
      ? t - this.samples[0][0]
      : 0;

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
// the safety index
// ----------------------------------------------------------------------

function severity(condition, state, absentDuration) {
  const span = SEVERITY_SPANS[condition];
  if (!span) return 1.0;

  let value;
  if (condition === "microsleep") value = state.blink?.closed_duration;
  else if (condition === "drowsy" || condition === "tired") value = state.drowsiness?.perclos;
  else if (condition === "distracted") value = state.distraction?.off_road_duration;
  else if (condition === "frequent_yawning") value = state.yawn?.recent_yawns;
  else if (condition === "driver_absent") value = absentDuration;

  if (value === null || value === undefined) return 1.0;

  return CONFIG.SEVERITY_FLOOR +
         (1 - CONFIG.SEVERITY_FLOOR) * ramp(value, span[0], span[1]);
}

/** How much this measurement deserves to be trusted. An eye
 *  measurement taken before calibration finished, or on a face at a
 *  steep angle, is worth less than one taken head-on. */
function confidence(condition, state) {
  if (FATIGUE.has(condition) && condition !== "frequent_yawning") {
    const c = state.blink?.confidence ?? 0.5;
    const turned = Math.min(Math.abs(state.head_pose?.yaw ?? 0) / 60, 1);
    return clamp(c * (1 - 0.4 * turned), 0.2, 1);
  }
  return 1.0;
}

function circadian(hour) {
  for (const [start, end, mult] of CIRCADIAN) {
    if (hour >= start && hour < end) return mult;
  }
  return 1.0;
}

export class DriverSafetyIndex {
  constructor() {
    this.score = null;
    this.level = "SAFE";
    this.last = null;
    this.active = new Set();
    this.faceLostSince = null;
    this.contributions = [];
    this.events = [];
  }

  update(state, t, hour) {
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

    // A microsleep is already the worst thing the fatigue family can
    // say, so saying it twice would charge one cause two penalties.
    if (active.has("microsleep")) { active.delete("drowsy"); active.delete("tired"); }
    if (active.has("drowsy")) active.delete("tired");

    const contributions = [];
    let total = 0;

    for (const condition of active) {
      const base = RISK_PENALTIES[condition] ?? 0;
      const sev = severity(condition, state, absentDuration);
      const conf = confidence(condition, state);
      const ctx = FATIGUE.has(condition) ? circadian(hour) : 1.0;

      const penalty = base * sev * conf * ctx;
      total += penalty;
      contributions.push({ condition, penalty, sev, conf, ctx });
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
      if (active.has(name)) { alert = { name, message, severity: sev }; break; }
    }

    return {
      score: this.score,
      level: this.level,
      active,
      alert,
      events,
      contributions,
      absent_duration: absentDuration
    };
  }
}

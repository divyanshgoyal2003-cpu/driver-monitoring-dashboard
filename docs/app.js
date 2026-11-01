/*
 * Camera, models and the screen. The measuring lives in detectors.js
 * and vision.js; this file only feeds them frames and draws what comes
 * back.
 *
 * Three models can run here and only one of them loads by default. The
 * face landmarker is 3.6 MB and every detector that matters depends on
 * it; the hand landmarker (7.8 MB) and the object detector (7.3 MB) are
 * each worth a third of the download again, for one condition apiece,
 * and the driver is on a phone. So they are opt-in, and the intro says
 * what they cost.
 *
 * The heavy models are also staggered across frames, as they are on the
 * desktop: phone on frame 0 of each six, hands on frame 2, seatbelt on
 * frame 3. Spacing them 6 apart rather than 3 was measured at 12.6 ->
 * 18+ FPS, because at 3 every single frame carried one of them and no
 * frame was ever cheap. The eyes are sampled on every frame either way,
 * which is what the frame rate was starving.
 */

import {
  FilesetResolver, FaceLandmarker, HandLandmarker, ObjectDetector
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

import {
  CONFIG, CONDITIONS, RISK_PENALTIES,
  BlinkDetector, DrowsinessDetector, YawnDetector, HeadPoseEstimator,
  DistractionDetector, CameraMonitor, WheelMonitor, PhoneMonitor,
  SeatbeltMonitor, ExpressionAnalyzer, DriverSafetyIndex, DrivingContext,
  detectSeatbelt, faceBox, circadianBand
} from "./detectors.js";

import { WeatherService, WEATHER_MULTIPLIERS } from "./weather.js";
import { save as saveSession, load as loadSessions, clear as clearSessions }
  from "./history.js";

const WASM = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
const FACE_MODEL = "./face_landmarker.task";
const HAND_MODEL = "./hand_landmarker.task";
const OBJECT_MODEL = "./efficientdet_lite0.tflite";

// EfficientDet-Lite is trained on COCO, which has a phone class. That
// label is the only thing wanted out of eighty of them.
const PHONE_LABELS = new Set(["cell phone", "mobile phone", "telephone"]);

const $ = id => document.getElementById(id);

const el = {
  intro: $("intro"), start: $("start"), error: $("error"), notice: $("notice"),
  live: $("live"), video: $("video"), overlay: $("overlay"),
  banner: $("banner"), bannerText: $("banner-text"), bannerDur: $("banner-dur"),
  calib: $("calib"),
  score: $("score"), ring: $("ring-value"), level: $("level"),
  contrib: $("contrib"), fps: $("fps"),
  ctxClock: $("ctx-clock"), ctxClockM: $("ctx-clock-m"),
  ctxTask: $("ctx-task"), ctxTaskM: $("ctx-task-m"),
  ctxWeather: $("ctx-weather"), ctxWeatherM: $("ctx-weather-m"),
  ctxLight: $("ctx-light"), ctxLightM: $("ctx-light-m"),
  ctxFatigue: $("ctx-fatigue"), ctxAttention: $("ctx-attention"),
  controls: $("controls"), hour: $("hour"), weather: $("weather"),
  locate: $("locate"), weatherNote: $("weather-note"),
  landmarks: $("landmarks"), recal: $("recal"), stop: $("stop"),
  useHands: $("use-hands"), usePhone: $("use-phone"),
  report: $("report"), reportBody: $("report-body"), reportClose: $("report-close"),
  history: $("history"), historyClear: $("history-clear")
};

const cells = {};
for (const node of document.querySelectorAll(".cell")) {
  cells[node.dataset.cell] = {
    node, v: node.querySelector(".v"), s: node.querySelector(".s")
  };
}

// ----------------------------------------------------------------------
// state
// ----------------------------------------------------------------------

let blink, drowsiness, yawn, headPose, distraction;
let camera, wheel, phone, seatbelt, expression, safety;

/** Every detector is rebuilt at the start of each drive, so a second
 *  session does not inherit the first one's calibration or counts. */
function freshDetectors() {
  blink = new BlinkDetector();
  drowsiness = new DrowsinessDetector();
  yawn = new YawnDetector();
  headPose = new HeadPoseEstimator();
  distraction = new DistractionDetector();
  camera = new CameraMonitor();
  wheel = new WheelMonitor();
  phone = new PhoneMonitor();
  seatbelt = new SeatbeltMonitor();
  expression = new ExpressionAnalyzer();
  safety = new DriverSafetyIndex();
}

freshDetectors();

const weather = new WeatherService();

let landmarker = null, hands = null, objects = null;
let stream = null;
let running = false;
let showLandmarks = true;
let lastVideoTime = -1;
let frameIndex = 0;
let frames = 0, fpsLast = 0, fps = 0;
let sessionStart = null, lastTick = null;
let wakeLock = null;
let hourOverride = null;
let declaredWeather = null;
let lastStrap = null;
let handPoints = null, phoneBoxes = null;

// Per-condition seconds and the score trace, for the end-of-drive
// report.
let tally = {};
let scoreTrace = [];
let lowest = 100;

// Small canvas for the whole-frame statistics, full-size one for the
// seatbelt's chest region. Both offscreen.
const sample = document.createElement("canvas");
sample.width = 64;
sample.height = 48;
const sampleCtx = sample.getContext("2d", { willReadFrequently: true });

const full = document.createElement("canvas");
const fullCtx = full.getContext("2d", { willReadFrequently: true });

// ----------------------------------------------------------------------
// sound
// ----------------------------------------------------------------------

let audio = null;
let lastAlertAt = 0;
let lastAlertName = null;

// A single steady beep is the easy thing to tune out, and the easy
// thing to sleep through -- the wrong property for the one alert that
// has to reach someone whose eyes have already closed. Both patterns
// pulse and rise, because interruption and rising pitch are what read
// as urgent; the rousing one climbs further and says it twice.
//
// ROUSE goes to the fatigue conditions, NOTIFY to the rest. A drowsy
// driver and a driver holding a phone are both "high" severity, but
// only one of them has to be woken up.
const ROUSE = [[1200, 160, 70], [1600, 160, 70], [2100, 420, 170],
               [1200, 160, 70], [1600, 160, 70], [2100, 520, 0]];
const NOTIFY = [[900, 140, 80], [1200, 260, 0]];

const FATIGUE_ALERTS = new Set(
  ["microsleep", "drowsy", "tired", "frequent_yawning"]);

function tone(frequency, ms, at) {
  if (!audio) return;
  const osc = audio.createOscillator();
  const gain = audio.createGain();
  osc.type = "square";
  osc.frequency.value = frequency;
  gain.gain.setValueAtTime(0.0001, at);
  gain.gain.exponentialRampToValueAtTime(0.25, at + 0.01);
  gain.gain.exponentialRampToValueAtTime(0.0001, at + ms / 1000);
  osc.connect(gain).connect(audio.destination);
  osc.start(at);
  osc.stop(at + ms / 1000 + 0.02);
}

function pattern(steps) {
  if (!audio) return;
  let at = audio.currentTime;
  for (const [frequency, ms, gap] of steps) {
    tone(frequency, ms, at);
    at += (ms + gap) / 1000;
  }
}

/** The spoken line matters more than the tone on a phone in a mount,
 *  so it carries the message. */
function alertFor(alert, t) {
  if (!alert) { lastAlertName = null; return; }

  // Each alert has to clear its own pattern (ROUSE runs about 2.0 s,
  // NOTIFY about 0.5 s) or it would simply run back to back: the
  // silence between is what stops it becoming background noise.
  const gap = alert.severity === "critical" ? 3.0 : 4.0;
  const fresh = alert.name !== lastAlertName;

  if (!fresh && t - lastAlertAt < gap) return;

  lastAlertAt = t;
  lastAlertName = alert.name;

  pattern(FATIGUE_ALERTS.has(alert.name) ? ROUSE : NOTIFY);

  // The spoken line, not the banner text: see SPOKEN in detectors.js
  // for why those are two different strings.
  if (fresh && alert.spoken && "speechSynthesis" in window) {
    const say = new SpeechSynthesisUtterance(alert.spoken);
    say.rate = 1.05;
    speechSynthesis.cancel();
    speechSynthesis.speak(say);
  }
}

// ----------------------------------------------------------------------
// frame statistics
// ----------------------------------------------------------------------

/**
 * Mean luminance, its spread, and a digest of the pixels.
 *
 * All three come off a 64x48 copy, which is far more than enough for
 * any of them. The spread is what tells a covered lens from a dark
 * cabin; the digest is what tells a frozen feed from a live one, since
 * a sensor's own noise means two consecutive frames are never bit
 * identical.
 */
function statistics(video) {
  sampleCtx.drawImage(video, 0, 0, sample.width, sample.height);

  let pixels;
  try {
    pixels = sampleCtx.getImageData(0, 0, sample.width, sample.height).data;
  } catch (e) {
    return null;    // a tainted canvas: nothing here can be measured
  }

  let sum = 0, sumSquares = 0;
  const count = sample.width * sample.height;

  // FNV-1a over the bytes. Any cheap digest would do; what matters is
  // that two different pictures almost never collide.
  let digest = 0x811c9dc5;

  for (let i = 0, p = 0; i < count; i++, p += 4) {
    const luma = 0.299 * pixels[p] + 0.587 * pixels[p + 1] + 0.114 * pixels[p + 2];
    sum += luma;
    sumSquares += luma * luma;

    digest ^= pixels[p];
    digest = (digest * 0x01000193) >>> 0;
    digest ^= pixels[p + 1];
    digest = (digest * 0x01000193) >>> 0;
    digest ^= pixels[p + 2];
    digest = (digest * 0x01000193) >>> 0;
  }

  const mean = sum / count;

  return {
    brightness: mean,
    variance: Math.sqrt(Math.max(0, sumSquares / count - mean * mean)),
    fingerprint: digest
  };
}

// ----------------------------------------------------------------------
// drawing
// ----------------------------------------------------------------------

const EYE_RING = [33, 160, 158, 133, 153, 144];
const EYE_RING_R = [362, 385, 387, 263, 373, 380];
const MOUTH_RING = [61, 13, 291, 14];

function draw(pts, state, shownHands, shownPhones) {
  const c = el.overlay;
  const ctx = c.getContext("2d");

  ctx.clearRect(0, 0, c.width, c.height);
  if (!showLandmarks) return;

  const style = getComputedStyle(document.documentElement);
  const ok = style.getPropertyValue("--ok").trim() || "#41c464";
  const bad = style.getPropertyValue("--bad").trim() || "#ff5f52";
  const warn = style.getPropertyValue("--warn").trim() || "#e5a234";
  const accent = style.getPropertyValue("--accent").trim() || "#4ea1ff";

  const ring = (idx, colour, source) => {
    ctx.beginPath();
    idx.forEach((i, n) => {
      const p = source[i];
      n === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y);
    });
    ctx.closePath();
    ctx.strokeStyle = colour;
    ctx.lineWidth = Math.max(2, c.width / 320);
    ctx.stroke();
  };

  if (pts) {
    const closed = state.blink?.eyes_closed;
    const yawning = (state.yawn?.mar ?? 0) > CONFIG.MAR_THRESHOLD;
    const offRoad = state.distraction && !state.distraction.eyes_on_road;

    ring(EYE_RING, closed ? bad : ok, pts);
    ring(EYE_RING_R, closed ? bad : ok, pts);
    ring(MOUTH_RING, yawning ? warn : ok, pts);

    // Where the head is pointing, as a line from the nose.
    const nose = pts[1];
    const len = c.width * 0.22;

    ctx.beginPath();
    ctx.moveTo(nose.x, nose.y);
    ctx.lineTo(
      nose.x + Math.sin((state.head_pose?.yaw ?? 0) * Math.PI / 180) * len,
      nose.y + Math.sin((state.head_pose?.pitch ?? 0) * Math.PI / 180) * len
    );
    ctx.strokeStyle = offRoad ? bad : accent;
    ctx.lineWidth = Math.max(2, c.width / 260);
    ctx.stroke();
  }

  // The chest region whenever it was looked at, so an UNKNOWN seatbelt
  // verdict is visibly about framing rather than about the belt -- and
  // the strap itself when one was found.
  const observation = state.seatbelt?.observation;

  if (observation?.roi) {
    const [x1, y1, x2, y2] = observation.roi;
    ctx.strokeStyle = observation.belt_found ? ok : warn;
    ctx.lineWidth = Math.max(1, c.width / 480);
    ctx.setLineDash([6, 6]);
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    ctx.setLineDash([]);
  }

  if (lastStrap) {
    ctx.strokeStyle = ok;
    ctx.lineWidth = Math.max(3, c.width / 200);
    for (const [x1, y1, x2, y2] of lastStrap) {
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.stroke();
    }
  }

  if (shownHands?.length) {
    ctx.fillStyle = state.wheel?.hands_off ? bad : ok;
    const radius = Math.max(3, c.width / 180);
    for (const hand of shownHands) {
      for (const p of hand) {
        ctx.beginPath();
        ctx.arc(p.x * c.width, p.y * c.height, radius, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }

  if (shownPhones?.length) {
    ctx.strokeStyle = bad;
    ctx.fillStyle = bad;
    ctx.lineWidth = Math.max(2, c.width / 240);
    ctx.font = `${Math.max(12, Math.round(c.width / 32))}px sans-serif`;
    for (const box of shownPhones) {
      ctx.strokeRect(box.originX, box.originY, box.width, box.height);
      ctx.fillText("phone", box.originX, Math.max(14, box.originY - 6));
    }
  }
}

// ----------------------------------------------------------------------
// screen
// ----------------------------------------------------------------------

const RING_CIRCUMFERENCE = 2 * Math.PI * 45;

function setCell(key, value, sub, state) {
  const cell = cells[key];
  if (!cell) return;
  cell.v.textContent = value;
  if (sub !== undefined) cell.s.textContent = sub;
  cell.node.dataset.state = state || "";
}

const ctxRow = key => document.querySelector(`.ctx-row[data-ctx="${key}"]`);

/**
 * The context layer, with its inputs on screen rather than only its
 * output.
 *
 * This panel exists because the previous version printed one line, and
 * only when something was amplifying. The hour was read from the
 * device every frame and the forecast was fetched and interpreted
 * correctly -- and a working forecast reporting a clear sky looked
 * exactly like a forecast that had never arrived, because both come
 * out as x1.00 against the words "nothing amplifying". Every row here
 * draws whether or not it is doing anything, and says where its number
 * came from.
 */
function renderContext(ctx) {
  const show = (row, value, multiplier, state) => {
    row.v.textContent = value;
    row.m.textContent = `×${multiplier.toFixed(2)}`;
    row.node.dataset.live = state ?? (multiplier > 1.001 ? "on" : "");
  };

  const rows = {
    clock: { node: ctxRow("clock"), v: el.ctxClock, m: el.ctxClockM },
    task: { node: ctxRow("task"), v: el.ctxTask, m: el.ctxTaskM },
    weather: { node: ctxRow("weather"), v: el.ctxWeather, m: el.ctxWeatherM },
    light: { node: ctxRow("light"), v: el.ctxLight, m: el.ctxLightM }
  };

  // The clock. `hourOverride` is the demo switch; when it is off, this
  // is simply what the device says the time is.
  const band = circadianBand(ctx.hour);
  const now = new Date();
  const real = `${String(now.getHours()).padStart(2, "0")}:` +
               `${String(now.getMinutes()).padStart(2, "0")}`;

  show(rows.clock,
    hourOverride === null
      ? `${real} · ${band.label}`
      : `${String(ctx.hour).padStart(2, "0")}:00 · ${band.label} (set by hand)`,
    band.multiplier);

  show(rows.task, clock(ctx.elapsed), ctx.timeOnTaskMultiplier());

  // The weather, and -- the part that was missing -- where it came
  // from and whether it actually arrived.
  const [condition, source] = currentWeather();
  const multiplier = ctx.weatherMultiplier();
  const reading = condition.replace(/_/g, " ");

  let text, state = null;

  if (source === "driver") {
    text = `${reading} · you said`;
  } else if (weather.status === "ready") {
    text = `${reading} · forecast` +
      (weather.temperature !== null ? ` · ${Math.round(weather.temperature)}°C` : "");
  } else if (weather.status === "locating") {
    text = "asking for your location…";
    state = "stale";
  } else if (weather.status === "fetching") {
    text = "fetching the forecast…";
    state = "stale";
  } else if (weather.status === "failed") {
    text = weather.error || "no reading";
    state = "bad";
  } else {
    text = "no reading — weighing nothing";
    state = "stale";
  }

  show(rows.weather, text, multiplier, state);

  show(rows.light,
    ctx.brightness === null
      ? "not measured"
      : `${ctx.brightness.toFixed(0)} of 255 · ` +
        (ctx.darkness > 0.66 ? "dark" : ctx.darkness > 0.2 ? "dim" : "well lit"),
    ctx.attentionMultiplier() / ctx.weatherMultiplier());

  const fatigue = ctx.fatigueMultiplier();
  const attention = ctx.attentionMultiplier();

  el.ctxFatigue.textContent = `×${fatigue.toFixed(2)}`;
  el.ctxAttention.textContent = `×${attention.toFixed(2)}`;
  el.ctxFatigue.dataset.live = fatigue > 1.001 ? "on" : "";
  el.ctxAttention.dataset.live = attention > 1.001 ? "on" : "";
}

function clock(seconds) {
  const mm = String(Math.floor(seconds / 60)).padStart(2, "0");
  const ss = String(Math.floor(seconds % 60)).padStart(2, "0");
  return `${mm}:${ss}`;
}

function render(state, risk, t) {
  const score = Math.round(risk.score);
  el.score.textContent = score;
  el.level.textContent = risk.level;
  el.level.dataset.level = risk.level;

  const colour = risk.level === "SAFE" ? "var(--ok)"
               : risk.level === "CAUTION" ? "var(--warn)" : "var(--bad)";

  el.ring.style.stroke = colour;
  el.ring.style.strokeDasharray = RING_CIRCUMFERENCE;
  el.ring.style.strokeDashoffset =
    RING_CIRCUMFERENCE * (1 - Math.max(0, Math.min(score, 100)) / 100);

  // What is costing points, largest first, each with its own three
  // layers written out -- so the arithmetic is on the screen and not
  // only in the README.
  if (!risk.contributions.length) {
    el.contrib.innerHTML = '<div class="none">nothing detected</div>';
  } else {
    el.contrib.innerHTML = risk.contributions.map(c =>
      `<div><span>${c.condition.replace(/_/g, " ")}</span>` +
      `<span class="why">${RISK_PENALTIES[c.condition]}` +
      ` &times;${c.sev.toFixed(2)} &times;${c.conf.toFixed(2)}` +
      `${c.ctx > 1.001 ? ` &times;${c.ctx.toFixed(2)}` : ""}</span>` +
      `<span class="cost">-${c.penalty.toFixed(1)}</span></div>`
    ).join("");
  }

  renderContext(risk.context);

  if (risk.alert) {
    el.banner.hidden = false;
    el.banner.dataset.sev = risk.alert.severity;
    el.bannerText.textContent = risk.alert.message;

    const dur =
      risk.alert.name === "distracted" ? state.distraction?.off_road_duration
    : risk.alert.name === "microsleep" ? state.blink?.closed_duration
    : risk.alert.name === "driver_absent" ? risk.absent_duration
    : risk.alert.name === "camera_blocked" ? state.camera?.duration
    : risk.alert.name === "phone" ? state.phone?.usage_duration
    : risk.alert.name === "hands_off" ? state.wheel?.missing_for
    : null;

    el.bannerDur.textContent = dur ? `${dur.toFixed(1)}s` : "";
  } else {
    el.banner.hidden = true;
  }

  const earCal = state.blink?.calibrated;
  const poseCal = state.head_pose?.calibrated;

  if (!poseCal || !earCal) {
    el.calib.hidden = false;
    el.calib.textContent = !poseCal
      ? "CALIBRATING HEAD POSE - look straight ahead"
      : "LEARNING YOUR BLINK - keep driving normally";
  } else {
    el.calib.hidden = true;
  }

  const b = state.blink, d = state.drowsiness, y = state.yawn,
        h = state.head_pose, x = state.distraction, cam = state.camera,
        w = state.wheel, ph = state.phone, sb = state.seatbelt,
        ex = state.emotion;

  if (b) {
    setCell("ear", b.ear.toFixed(3), `threshold ${b.threshold.toFixed(3)}`,
      b.eyes_closed ? "bad" : "ok");
    setCell("blinks", b.blink_count, `${b.blink_rate}/min`, "");
    setCell("microsleep", b.microsleep_count,
      b.closed_duration > 0.3 ? `closed ${b.closed_duration.toFixed(1)}s` : "episodes",
      b.microsleep ? "bad" : "");
  } else {
    setCell("ear", "--", "no face", "idle");
    setCell("microsleep", blink.microsleepCount, "episodes", "idle");
  }

  if (d) {
    setCell("perclos", d.ready ? `${(d.perclos * 100).toFixed(0)}%` : "--",
      d.ready ? d.level.toLowerCase() : `warming up ${d.span.toFixed(0)}s`,
      d.level === "DROWSY" ? "bad" : d.level === "TIRED" ? "warn" : "ok");
  }

  if (y) {
    setCell("yawns", y.yawn_count, `MAR ${y.mar.toFixed(2)}`,
      y.recent_yawns >= CONFIG.YAWN_ALERT_COUNT ? "warn" : "");
  }

  if (h) {
    setCell("yaw", `${h.yaw >= 0 ? "+" : ""}${h.yaw.toFixed(0)}°`,
      `pitch ${h.pitch >= 0 ? "+" : ""}${h.pitch.toFixed(0)}°`,
      !h.reliable ? "idle" : Math.abs(h.yaw) > CONFIG.OFF_ROAD_YAW ? "bad" : "ok");
  } else {
    setCell("yaw", "--", "no face", "idle");
  }

  setCell("distractions", distraction.count,
    x && x.off_road_duration > 0.2
      ? `off road ${x.off_road_duration.toFixed(1)}s` : "episodes",
    x?.distracted ? "bad" : "");

  if (cam) {
    setCell("camera",
      cam.blocked ? (cam.frozen ? "FROZEN" : "BLOCKED") : "clear",
      `light ${cam.brightness.toFixed(0)} · spread ${cam.variance.toFixed(1)}`,
      cam.blocked ? "bad" : "ok");
  }

  // "cannot see" is a real answer and the cell says it, because a
  // monitor that quietly reports zero is worse than one that admits
  // the wheel is out of frame.
  if (!el.useHands.checked) {
    setCell("wheel", "off", "model not loaded", "idle");
  } else if (w) {
    setCell("wheel",
      w.can_see ? (w.hands_off ? "OFF" : "on") : "cannot see",
      w.can_see
        ? (w.hands_off ? `empty ${w.missing_for.toFixed(1)}s`
                       : `${w.hand_count} at the wheel`)
        : `hands in ${(w.share * 100).toFixed(0)}% of window`,
      !w.can_see ? "idle" : w.hands_off ? "bad" : "ok");
  }

  if (!el.usePhone.checked) {
    setCell("phone", "off", "model not loaded", "idle");
  } else if (ph) {
    setCell("phone", ph.episodes,
      ph.phone_in_use ? `held ${ph.usage_duration.toFixed(1)}s` : "episodes",
      ph.phone_in_use ? "bad" : "");
  }

  if (sb) {
    setCell("seatbelt", sb.status.toLowerCase(),
      sb.status === "UNKNOWN" ? "chest not in view"
                              : `strap in ${(sb.ratio * 100).toFixed(0)}% of frames`,
      sb.status === "ON" ? "ok" : sb.status === "OFF" ? "warn" : "idle");
  }

  if (ex) {
    setCell("expression", `${(ex.confidence * 100).toFixed(0)}%`,
      ex.negative ? "negative" : "neutral",
      ex.negative ? "warn" : "ok");
  } else {
    setCell("expression", "--", "no face", "idle");
  }

  const elapsed = sessionStart === null ? 0 : t - sessionStart;
  setCell("session", clock(elapsed), `${fps.toFixed(0)} fps`, "");

  el.fps.textContent = `${fps.toFixed(0)} fps`;
}

// ----------------------------------------------------------------------
// the loop
// ----------------------------------------------------------------------

function tick() {
  if (!running) return;

  const video = el.video;
  const now = performance.now();
  const t = now / 1000;

  if (video.readyState >= 2 && video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;

    if (el.overlay.width !== video.videoWidth) {
      el.overlay.width = video.videoWidth;
      el.overlay.height = video.videoHeight;
      full.width = video.videoWidth;
      full.height = video.videoHeight;
    }

    let result = null;
    try {
      result = landmarker.detectForVideo(video, now);
    } catch (e) {
      // A dropped frame is not worth taking the session down for.
    }

    const faces = result?.faceLandmarks ?? [];
    const matrices = result?.facialTransformationMatrixes ?? [];
    const shapes = result?.faceBlendshapes ?? [];
    const found = faces.length > 0;

    const pts = found
      ? faces[0].map(p => ({
          x: p.x * video.videoWidth, y: p.y * video.videoHeight
        }))
      : null;

    const state = { face_detected: found };

    state.blink = blink.update(pts, t);
    state.drowsiness = drowsiness.update(state.blink, t);
    state.yawn = yawn.update(pts, t);
    state.head_pose = headPose.update(
      found && matrices.length ? matrices[0].data : null,
      state.yawn?.mar ?? null,
      t
    );
    state.distraction = distraction.update(state.head_pose, t);
    state.emotion = expression.update(shapes.length ? shapes[0].categories : null);

    const stats = statistics(video);
    if (stats) {
      state.camera = camera.update(
        stats.brightness, stats.variance, stats.fingerprint, t);
    }

    const box = pts ? faceBox(pts) : null;

    // The heavy work, staggered across each group of six frames.
    const slot = frameIndex % CONFIG.PHONE_DETECTION_INTERVAL;

    if (objects && slot === 0) phoneBoxes = detectPhones(video, now);

    if (hands && slot === 2) {
      try {
        const detected = hands.detectForVideo(video, now);
        handPoints = detected?.landmarks ?? [];
      } catch (e) {
        handPoints = handPoints ?? [];
      }
    }

    if (slot === 3 && box) {
      fullCtx.drawImage(video, 0, 0);
      const observation = detectSeatbelt(
        fullCtx, box, video.videoWidth, video.videoHeight);
      state.seatbelt = seatbelt.update(observation, t);
      lastStrap = observation.strap;
    } else {
      state.seatbelt = seatbelt.update(null, t);
    }

    // The wheel line is the bottom of the face box: a hand above the
    // chin is up by the face, not on a wheel.
    const wheelLine = box ? box[3] / video.videoHeight : 0.6;

    state.wheel = wheel.update(hands ? (handPoints ?? []) : null, wheelLine, t);
    state.phone = phone.update(
      objects ? (phoneBoxes ?? []).map(b => ({ confidence: b.score })) : null, t);

    const [condition, source] = currentWeather();

    const context = new DrivingContext({
      hour: hourOverride === null ? new Date().getHours() : hourOverride,
      elapsed: sessionStart === null ? 0 : t - sessionStart,
      brightness: stats ? stats.brightness : null,
      weather: condition,
      weatherSource: source
    });

    const risk = safety.update(state, t, context);

    accumulate(risk, t);
    alertFor(risk.alert, t);
    draw(pts, state, hands ? handPoints : null, objects ? phoneBoxes : null);
    render(state, risk, t);

    // Rate limited inside the service, so calling it every frame only
    // actually fetches every fifteen minutes.
    weather.refresh(t).then(tried => { if (tried) showWeather(); });

    frameIndex += 1;
    frames += 1;

    if (now - fpsLast > 500) {
      fps = frames * 1000 / (now - fpsLast);
      frames = 0;
      fpsLast = now;
    }
  }

  requestAnimationFrame(tick);
}

function detectPhones(video, now) {
  try {
    const detected = objects.detectForVideo(video, now);

    return (detected?.detections ?? [])
      .map(d => ({
        box: d.boundingBox,
        label: d.categories?.[0]?.categoryName ?? "",
        score: d.categories?.[0]?.score ?? 0
      }))
      .filter(d => d.box &&
                   PHONE_LABELS.has(d.label.toLowerCase()) &&
                   d.score >= CONFIG.PHONE_CONFIDENCE)
      .map(d => ({
        originX: d.box.originX, originY: d.box.originY,
        width: d.box.width, height: d.box.height, score: d.score
      }));
  } catch (e) {
    return phoneBoxes ?? [];
  }
}

/** Seconds each condition was active, and where the score went. */
function accumulate(risk, t) {
  // Clamped, so a tab that was in the background for a minute does not
  // bill the whole minute to whatever was active when it left.
  const dt = lastTick === null ? 0 : Math.min(t - lastTick, 1);
  lastTick = t;

  for (const condition of risk.active) {
    tally[condition] = (tally[condition] ?? 0) + dt;
  }

  if (risk.score < lowest) lowest = risk.score;

  const elapsed = sessionStart === null ? 0 : t - sessionStart;

  // One sample a second is plenty for a trace a thumb's width wide.
  if (!scoreTrace.length ||
      elapsed - scoreTrace[scoreTrace.length - 1][0] >= 1) {
    scoreTrace.push([elapsed, risk.score]);
  }
}

// ----------------------------------------------------------------------
// weather
// ----------------------------------------------------------------------

/** The driver's word beats the forecast, and the forecast beats
 *  nothing. A 1.0 from a clear sky is not the same evidence as a 1.0
 *  from ignorance, so the source travels with the reading. */
function currentWeather() {
  if (declaredWeather !== null) return [declaredWeather, "driver"];
  const [observed, source] = weather.provenance();
  return [observed ?? "clear", source];
}

function showWeather() {
  const [condition, source] = currentWeather();
  const multiplier = WEATHER_MULTIPLIERS[condition] ?? 1.0;

  const where = source === "driver" ? "you said"
              : weather.status === "ready" ? "forecast"
              : weather.status === "locating" ? "asking for your location"
              : weather.status === "fetching" ? "fetching"
              : weather.status === "failed" ? (weather.error || "no reading")
              : "no reading";

  el.weatherNote.textContent =
    `${condition.replace(/_/g, " ")} ×${multiplier.toFixed(2)} · ${where}`;

  // The panel is the real display; keep it in step whenever this
  // changes, including while nothing is running.
  if (!running) {
    renderContext(new DrivingContext({
      hour: hourOverride === null ? new Date().getHours() : hourOverride,
      weather: condition,
      weatherSource: source
    }));
  }
}

/**
 * Ask for a location and a forecast without the driver pressing
 * anything.
 *
 * Deliberately not awaited by start(): a forecast is worth having and
 * is worth nothing at all compared to the camera already running, so
 * it must never be able to delay or fail the drive. If the browser
 * prompts, it prompts over a monitor that is already working.
 */
function fetchWeather() {
  weather.begin(performance.now() / 1000)
    .then(showWeather)
    .catch(() => showWeather());

  showWeather();
}

// ----------------------------------------------------------------------
// the end-of-drive report
// ----------------------------------------------------------------------

function trace(points, width = 280, height = 48) {
  if (!points || points.length < 2) return "";

  const span = points[points.length - 1][0] || 1;

  const path = points.map(([at, score], i) => {
    const x = (at / span) * width;
    const y = height - (Math.max(0, Math.min(100, score)) / 100) * height;
    return `${i === 0 ? "M" : "L"}${x.toFixed(1)} ${y.toFixed(1)}`;
  }).join(" ");

  // The two band edges, so a glance says which band the trace is in.
  const at = v => (height - (v / 100) * height).toFixed(1);

  return `<svg class="trace" viewBox="0 0 ${width} ${height}" role="img"
    aria-label="safety score over the drive">
    <line class="band" x1="0" y1="${at(CONFIG.SAFE_SCORE)}"
          x2="${width}" y2="${at(CONFIG.SAFE_SCORE)}"/>
    <line class="band" x1="0" y1="${at(CONFIG.CAUTION_SCORE)}"
          x2="${width}" y2="${at(CONFIG.CAUTION_SCORE)}"/>
    <path d="${path}"/>
  </svg>`;
}

function buildReport(t) {
  const counted = CONDITIONS
    .map(([name]) => [name, tally[name] ?? 0])
    .filter(([, seconds]) => seconds >= 0.5)
    .sort((a, b) => b[1] - a[1]);

  return {
    at: Date.now(),
    duration: sessionStart === null ? 0 : t - sessionStart,
    lowest,
    blinks: blink.blinkCount,
    microsleeps: blink.microsleepCount,
    yawns: yawn.yawnCount,
    distractions: distraction.count,
    phone: phone.episodes,
    hands_off: wheel.episodes,
    seatbelt: seatbelt.status,
    conditions: counted,
    // Halved before storing: twenty drives of one sample a second
    // would eventually be the thing that fills the quota.
    trace: scoreTrace.filter((_, i) => i % 2 === 0)
  };
}

function renderReport(report) {
  const rows = report.conditions.length
    ? report.conditions.map(([name, seconds]) =>
        `<div><span>${name.replace(/_/g, " ")}</span>` +
        `<span class="cost">${seconds.toFixed(1)}s</span></div>`).join("")
    : '<div class="none">nothing was detected for as long as half a second</div>';

  el.reportBody.innerHTML = `
    <div class="figures">
      <div><strong>${clock(report.duration)}</strong><span>at the wheel</span></div>
      <div><strong>${Math.round(report.lowest)}</strong><span>lowest score</span></div>
      <div><strong>${report.microsleeps}</strong><span>microsleeps</span></div>
      <div><strong>${report.distractions}</strong><span>distractions</span></div>
      <div><strong>${report.blinks}</strong><span>blinks</span></div>
      <div><strong>${report.yawns}</strong><span>yawns</span></div>
    </div>
    ${trace(report.trace)}
    <h3>Time spent in each condition</h3>
    <div class="contrib">${rows}</div>
    <p class="note" style="margin-top:14px">
      The seatbelt ended as <strong>${report.seatbelt.toLowerCase()}</strong>.
      ${report.seatbelt === "UNKNOWN"
        ? "The chest was not in view often enough to judge, which is the " +
          "honest answer rather than a pass."
        : ""}
    </p>`;

  renderHistory();
}

function renderHistory() {
  const saved = loadSessions();

  if (!saved.length) {
    el.history.innerHTML = '<div class="none">no earlier drives stored</div>';
    return;
  }

  el.history.innerHTML = saved.map(drive => {
    const when = new Date(drive.at);
    const band = drive.lowest >= CONFIG.SAFE_SCORE ? "ok"
               : drive.lowest >= CONFIG.CAUTION_SCORE ? "warn" : "bad";

    return `<div class="drive">
      <span>${when.toLocaleDateString()} ${when.toLocaleTimeString([],
        { hour: "2-digit", minute: "2-digit" })}</span>
      <span>${clock(drive.duration)}</span>
      <span data-band="${band}">low ${Math.round(drive.lowest)}</span>
      <span>${drive.microsleeps} microsleep · ${drive.distractions} distraction</span>
    </div>`;
  }).join("");
}

// ----------------------------------------------------------------------
// start / stop
// ----------------------------------------------------------------------

function fail(message, detail) {
  el.error.hidden = false;
  el.error.innerHTML = `<strong>${message}</strong>` +
    (detail ? `<br><span style="opacity:.8">${detail}</span>` : "");
  el.start.disabled = false;
  el.start.textContent = "Try again";
}

async function start() {
  el.error.hidden = true;
  el.report.hidden = true;
  el.notice.hidden = true;
  el.start.disabled = true;

  if (!window.isSecureContext) {
    return fail("This page needs HTTPS.",
      "Browsers only give the camera to secure pages. Open the https:// " +
      "address, or use localhost.");
  }

  el.start.textContent = "Starting the camera…";

  try {
    audio = new (window.AudioContext || window.webkitAudioContext)();
    if (audio.state === "suspended") await audio.resume();
  } catch (e) {
    audio = null;
  }

  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } },
      audio: false
    });
  } catch (e) {
    return fail("The camera was refused.",
      `${e.name}: allow camera access for this site and try again.`);
  }

  el.video.srcObject = stream;
  await el.video.play();

  el.start.textContent = "Loading the face model…";

  let fileset;

  try {
    fileset = await FilesetResolver.forVisionTasks(WASM);
    landmarker = await FaceLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: FACE_MODEL, delegate: "GPU" },
      runningMode: "VIDEO",
      numFaces: 1,
      outputFacialTransformationMatrixes: true,
      outputFaceBlendshapes: true
    });
  } catch (e) {
    return fail("The face model would not load.", String(e.message || e));
  }

  // The optional models. A failure here costs one condition, not the
  // session, so it says so in the notice line and carries on.
  const missing = [];

  if (el.useHands.checked && !hands) {
    el.start.textContent = "Loading the hand model… (7.8 MB)";
    try {
      hands = await HandLandmarker.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: HAND_MODEL, delegate: "GPU" },
        runningMode: "VIDEO",
        numHands: 2,
        minHandDetectionConfidence: CONFIG.HAND_CONFIDENCE
      });
    } catch (e) {
      hands = null;
      missing.push("hands on the wheel");
    }
  }

  if (el.usePhone.checked && !objects) {
    el.start.textContent = "Loading the object model… (7.3 MB)";
    try {
      objects = await ObjectDetector.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: OBJECT_MODEL, delegate: "GPU" },
        runningMode: "VIDEO",
        scoreThreshold: CONFIG.PHONE_CONFIDENCE,
        maxResults: 8
      });
    } catch (e) {
      objects = null;
      missing.push("phone in hand");
    }
  }

  if (missing.length) {
    el.notice.hidden = false;
    el.notice.textContent =
      `Running without ${missing.join(" and ")}: the model would not load.`;
  }

  try {
    wakeLock = await navigator.wakeLock?.request("screen");
  } catch (e) { /* optional */ }

  freshDetectors();
  tally = {};
  scoreTrace = [];
  lowest = 100;
  lastTick = null;
  frameIndex = 0;
  handPoints = null;
  phoneBoxes = null;
  lastStrap = null;

  sessionStart = performance.now() / 1000;
  fpsLast = performance.now();
  running = true;

  el.intro.hidden = true;
  el.live.hidden = false;
  el.controls.hidden = false;

  // Coordinates are remembered after the first time, so on every drive
  // but the first this is silent and the forecast is simply there.
  fetchWeather();
  requestAnimationFrame(tick);
}

function stop() {
  if (!running) return;

  running = false;

  const report = buildReport(performance.now() / 1000);

  stream?.getTracks().forEach(track => track.stop());
  wakeLock?.release?.();
  wakeLock = null;
  speechSynthesis?.cancel?.();

  // A drive under ten seconds is a mis-click, not a session.
  if (report.duration >= 10) saveSession(report);

  renderReport(report);

  el.live.hidden = true;
  el.controls.hidden = true;
  el.report.hidden = false;
  el.start.disabled = false;
  el.start.textContent = "Start monitoring";
}

// ----------------------------------------------------------------------
// wiring
// ----------------------------------------------------------------------

el.start.addEventListener("click", start);
el.stop.addEventListener("click", stop);

el.reportClose.addEventListener("click", () => {
  el.report.hidden = true;
  el.intro.hidden = false;
});

el.historyClear.addEventListener("click", () => {
  clearSessions();
  renderHistory();
});

el.recal.addEventListener("click", () => {
  blink.recalibrate();
  headPose.startCalibration();
});

el.landmarks.addEventListener("click", () => {
  showLandmarks = !showLandmarks;
  el.landmarks.textContent = showLandmarks ? "Hide landmarks" : "Show landmarks";
});

el.hour.addEventListener("change", () => {
  hourOverride = el.hour.value === "" ? null : Number(el.hour.value);
  // The running loop redraws the panel every frame, but a demo is
  // usually set up before pressing start, and until this the panel
  // simply ignored the selector while idle.
  showWeather();
});

el.weather.addEventListener("change", () => {
  declaredWeather = el.weather.value === "" ? null : el.weather.value;
  showWeather();
});

// Only a retry now, for a declined prompt or a forecast that did not
// arrive. The normal path asks for nothing.
el.locate.addEventListener("click", async () => {
  el.locate.disabled = true;
  el.locate.textContent = "Asking…";

  // A fresh attempt, so a driver who declined and changed their mind
  // is not held to the stored refusal.
  weather.fetchedAt = null;
  const ok = await weather.begin(performance.now() / 1000);

  el.locate.disabled = false;
  el.locate.textContent = ok ? "Refresh forecast" : "Retry forecast";
  showWeather();
});

// Loading a model takes seconds, and doing it mid-drive would stall the
// loop, so a checkbox only takes effect on the next start and says so.
for (const box of [el.useHands, el.usePhone]) {
  box.addEventListener("change", () => {
    if (!running) return;
    el.notice.hidden = false;
    el.notice.textContent = "That takes effect the next time you start.";
  });
}

// The screen sleeping mid-drive would be an odd way to monitor a
// driver, so the lock is taken again whenever the tab comes back.
document.addEventListener("visibilitychange", async () => {
  if (document.visibilityState === "visible" && running && !wakeLock) {
    try { wakeLock = await navigator.wakeLock?.request("screen"); } catch (e) {}
  }
});

renderHistory();
showWeather();

// If this device has been located before, the coordinates are already
// in hand and the forecast can be on screen before the driver has
// pressed anything at all. No prompt is possible on this path, because
// nothing is being asked for.
if (weather.located) {
  weather.refresh(performance.now() / 1000).then(showWeather).catch(() => {});
}

if ("serviceWorker" in navigator && location.protocol === "https:") {
  navigator.serviceWorker.register("./sw.js").catch(() => {});
}

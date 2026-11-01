/*
 * Camera, landmarks and the screen. The measuring lives in
 * detectors.js; this file only feeds it frames and draws what comes
 * back.
 */

import {
  FilesetResolver, FaceLandmarker
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

import {
  CONFIG, BlinkDetector, DrowsinessDetector, YawnDetector,
  HeadPoseEstimator, DistractionDetector, DriverSafetyIndex
} from "./detectors.js";

const WASM = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
const MODEL = "./face_landmarker.task";

const $ = id => document.getElementById(id);

const el = {
  intro: $("intro"), start: $("start"), error: $("error"),
  live: $("live"), video: $("video"), overlay: $("overlay"),
  banner: $("banner"), bannerText: $("banner-text"), bannerDur: $("banner-dur"),
  calib: $("calib"),
  score: $("score"), ring: $("ring-value"), level: $("level"),
  contrib: $("contrib"), fps: $("fps"),
  controls: $("controls"), hour: $("hour"),
  landmarks: $("landmarks"), recal: $("recal"), stop: $("stop")
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

const blink = new BlinkDetector();
const drowsiness = new DrowsinessDetector();
const yawn = new YawnDetector();
const headPose = new HeadPoseEstimator();
const distraction = new DistractionDetector();
const safety = new DriverSafetyIndex();

let landmarker = null;
let stream = null;
let running = false;
let showLandmarks = true;
let lastVideoTime = -1;
let frames = 0, fpsLast = 0, fps = 0;
let sessionStart = null;
let wakeLock = null;
let hourOverride = null;

// ----------------------------------------------------------------------
// sound
// ----------------------------------------------------------------------

let audio = null;
let lastAlertAt = 0;
let lastAlertName = null;

function beep(frequency, ms) {
  if (!audio) return;
  const osc = audio.createOscillator();
  const gain = audio.createGain();
  osc.type = "square";
  osc.frequency.value = frequency;
  gain.gain.setValueAtTime(0.0001, audio.currentTime);
  gain.gain.exponentialRampToValueAtTime(0.25, audio.currentTime + 0.01);
  gain.gain.exponentialRampToValueAtTime(0.0001, audio.currentTime + ms / 1000);
  osc.connect(gain).connect(audio.destination);
  osc.start();
  osc.stop(audio.currentTime + ms / 1000 + 0.02);
}

/** Critical conditions repeat urgently; the rest get two pulses. The
 *  spoken line matters more than the tone on a phone in a pocket
 *  mount, so it carries the message. */
function alertFor(alert, t) {
  if (!alert) { lastAlertName = null; return; }

  const gap = alert.severity === "critical" ? 1.2 : 4.0;
  const fresh = alert.name !== lastAlertName;

  if (!fresh && t - lastAlertAt < gap) return;

  lastAlertAt = t;
  lastAlertName = alert.name;

  if (alert.severity === "critical") {
    beep(1500, 160); setTimeout(() => beep(1500, 160), 200);
  } else {
    beep(900, 120); setTimeout(() => beep(1200, 120), 160);
  }

  if (fresh && "speechSynthesis" in window) {
    const say = new SpeechSynthesisUtterance(alert.message.replace(/-/g, ","));
    say.rate = 1.1;
    speechSynthesis.cancel();
    speechSynthesis.speak(say);
  }
}

// ----------------------------------------------------------------------
// drawing
// ----------------------------------------------------------------------

const EYE_RING = [33, 160, 158, 133, 153, 144];
const EYE_RING_R = [362, 385, 387, 263, 373, 380];
const MOUTH_RING = [61, 13, 291, 14];

function draw(pts, state) {
  const c = el.overlay;
  const ctx = c.getContext("2d");

  ctx.clearRect(0, 0, c.width, c.height);
  if (!pts || !showLandmarks) return;

  const style = getComputedStyle(document.documentElement);
  const closed = state.blink?.eyes_closed;
  const yawning = (state.yawn?.mar ?? 0) > CONFIG.MAR_THRESHOLD;
  const offRoad = state.distraction && !state.distraction.eyes_on_road;

  const ring = (idx, colour) => {
    ctx.beginPath();
    idx.forEach((i, n) => {
      const p = pts[i];
      n === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y);
    });
    ctx.closePath();
    ctx.strokeStyle = colour;
    ctx.lineWidth = Math.max(2, c.width / 320);
    ctx.stroke();
  };

  const ok = style.getPropertyValue("--ok").trim() || "#41c464";
  const bad = style.getPropertyValue("--bad").trim() || "#ff5f52";
  const warn = style.getPropertyValue("--warn").trim() || "#e5a234";

  ring(EYE_RING, closed ? bad : ok);
  ring(EYE_RING_R, closed ? bad : ok);
  ring(MOUTH_RING, yawning ? warn : ok);

  // Where the head is pointing, as a line from the nose.
  const nose = pts[1];
  const yawDeg = state.head_pose?.yaw ?? 0;
  const pitchDeg = state.head_pose?.pitch ?? 0;
  const len = c.width * 0.22;

  ctx.beginPath();
  ctx.moveTo(nose.x, nose.y);
  ctx.lineTo(
    nose.x + Math.sin(yawDeg * Math.PI / 180) * len,
    nose.y + Math.sin(pitchDeg * Math.PI / 180) * len
  );
  ctx.strokeStyle = offRoad ? bad : style.getPropertyValue("--accent").trim();
  ctx.lineWidth = Math.max(2, c.width / 260);
  ctx.stroke();
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

function render(state, risk, t) {
  // score ring
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

  // what is costing points, largest first
  if (!risk.contributions.length) {
    el.contrib.innerHTML = '<div class="none">nothing detected</div>';
  } else {
    el.contrib.innerHTML = risk.contributions.map(c =>
      `<div><span>${c.condition.replace(/_/g, " ")}</span>` +
      `<span>-${c.penalty.toFixed(1)}</span></div>`
    ).join("");
  }

  // banner
  if (risk.alert) {
    el.banner.hidden = false;
    el.banner.dataset.sev = risk.alert.severity;
    el.bannerText.textContent = risk.alert.message;

    const dur = risk.alert.name === "distracted"
      ? state.distraction?.off_road_duration
      : risk.alert.name === "microsleep"
        ? state.blink?.closed_duration
        : risk.alert.name === "driver_absent"
          ? risk.absent_duration
          : null;

    el.bannerDur.textContent = dur ? `${dur.toFixed(1)}s` : "";
  } else {
    el.banner.hidden = true;
  }

  // calibration notice
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

  // metrics
  const b = state.blink, d = state.drowsiness, y = state.yawn,
        h = state.head_pose, x = state.distraction;

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
    x && x.off_road_duration > 0.2 ? `off road ${x.off_road_duration.toFixed(1)}s` : "episodes",
    x?.distracted ? "bad" : "");

  const elapsed = sessionStart === null ? 0 : t - sessionStart;
  const mm = String(Math.floor(elapsed / 60)).padStart(2, "0");
  const ss = String(Math.floor(elapsed % 60)).padStart(2, "0");
  setCell("session", `${mm}:${ss}`, `${fps.toFixed(0)} fps`, "");

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
    }

    let result = null;
    try {
      result = landmarker.detectForVideo(video, now);
    } catch (e) {
      // A dropped frame is not worth taking the session down for.
    }

    const faces = result?.faceLandmarks ?? [];
    const matrices = result?.facialTransformationMatrixes ?? [];
    const found = faces.length > 0;

    const pts = found
      ? faces[0].map(p => ({ x: p.x * video.videoWidth, y: p.y * video.videoHeight }))
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

    const hour = hourOverride === null ? new Date().getHours() : hourOverride;
    const risk = safety.update(state, t, hour);

    alertFor(risk.alert, t);
    draw(pts, state);
    render(state, risk, t);

    frames += 1;
    if (now - fpsLast > 500) {
      fps = frames * 1000 / (now - fpsLast);
      frames = 0;
      fpsLast = now;
    }
  }

  requestAnimationFrame(tick);
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
  el.start.disabled = true;
  el.start.textContent = "Starting the camera…";

  if (!window.isSecureContext) {
    return fail("This page needs HTTPS.",
      "Browsers only give the camera to secure pages. Open the https:// address, or use localhost.");
  }

  try {
    audio = new (window.AudioContext || window.webkitAudioContext)();
    if (audio.state === "suspended") await audio.resume();
  } catch (e) { audio = null; }

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

  try {
    const fileset = await FilesetResolver.forVisionTasks(WASM);
    landmarker = await FaceLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: MODEL, delegate: "GPU" },
      runningMode: "VIDEO",
      numFaces: 1,
      outputFacialTransformationMatrixes: true
    });
  } catch (e) {
    return fail("The face model would not load.", String(e.message || e));
  }

  try {
    wakeLock = await navigator.wakeLock?.request("screen");
  } catch (e) { /* optional */ }

  sessionStart = performance.now() / 1000;
  fpsLast = performance.now();
  running = true;

  el.intro.hidden = true;
  el.live.hidden = false;
  el.controls.hidden = false;

  requestAnimationFrame(tick);
}

function stop() {
  running = false;
  stream?.getTracks().forEach(track => track.stop());
  wakeLock?.release?.();
  wakeLock = null;
  speechSynthesis?.cancel?.();

  el.live.hidden = true;
  el.controls.hidden = true;
  el.intro.hidden = false;
  el.start.disabled = false;
  el.start.textContent = "Start monitoring";
}

// ----------------------------------------------------------------------
// wiring
// ----------------------------------------------------------------------

el.start.addEventListener("click", start);
el.stop.addEventListener("click", stop);

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
});

// The screen sleeping mid-drive would be an odd way to monitor a
// driver, so the lock is taken again whenever the tab comes back.
document.addEventListener("visibilitychange", async () => {
  if (document.visibilityState === "visible" && running && !wakeLock) {
    try { wakeLock = await navigator.wakeLock?.request("screen"); } catch (e) {}
  }
});

if ("serviceWorker" in navigator && location.protocol === "https:") {
  navigator.serviceWorker.register("./sw.js").catch(() => {});
}

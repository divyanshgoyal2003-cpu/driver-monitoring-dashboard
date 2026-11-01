/*
 * What the camera view draws on top of itself.
 *
 * Split out of app.js for two reasons. It is the only part of this
 * whose output is judged by eye rather than by assertion, so it needs
 * to be renderable against a still image without starting a camera,
 * a monitoring loop and three models. And app.js had grown past
 * fifteen hundred lines with the drawing wedged in the middle of it.
 *
 * Nothing here feeds a detector. This file draws; it does not decide.
 */

import { FaceLandmarker, HandLandmarker }
  from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

/*
 * This was six-point hexagons round the eyes and a four-point line
 * across the mouth, which is exactly what it looked like: debug
 * output. It was drawn from the landmarks the MEASUREMENTS need -- the
 * EAR wants six points per eye and the MAR four round the lips, so six
 * and four were what got drawn. Reasonable for checking the maths,
 * wrong for the only part of this anybody looks at for more than a
 * second.
 *
 * MediaPipe ships the real contours alongside the landmarks, so the
 * picture can be the proper shape of an eye while the maths behind it
 * stays exactly as it was.
 *
 * Two things keep it cheap. Every connection list is stroked as ONE
 * path, which is what makes a 2,556-segment mesh affordable at all,
 * and the mesh can be turned off on a device that cannot spare it.
 */

const F = FaceLandmarker;
const MESH  = F.FACE_LANDMARKS_TESSELATION ?? [];
const OVAL  = F.FACE_LANDMARKS_FACE_OVAL ?? [];
const EYES  = [...(F.FACE_LANDMARKS_LEFT_EYE ?? []),
               ...(F.FACE_LANDMARKS_RIGHT_EYE ?? [])];
const BROWS = [...(F.FACE_LANDMARKS_LEFT_EYEBROW ?? []),
               ...(F.FACE_LANDMARKS_RIGHT_EYEBROW ?? [])];
const LIPS  = F.FACE_LANDMARKS_LIPS ?? [];
const IRISES = [F.FACE_LANDMARKS_LEFT_IRIS ?? [],
                F.FACE_LANDMARKS_RIGHT_IRIS ?? []];

// The model may or may not return the iris points depending on build,
// and a hand model is only loaded if the driver ticked the box.
const HAND_BONES = HandLandmarker.HAND_CONNECTIONS ?? [];

/** mesh | outline | off -- cycled by the button, because a mesh is the
 *  first thing to give up on a phone that is dropping frames. */
let overlay = "mesh";

const OVERLAY_LABEL = {
  mesh: "Overlay: mesh",
  outline: "Overlay: outline",
  off: "Overlay: off"
};

function withAlpha(hex, a) {
  const h = hex.replace("#", "").trim();
  const full = h.length === 3 ? h.split("").map(x => x + x).join("") : h;
  const n = parseInt(full, 16);

  if (!Number.isFinite(n)) return hex;

  return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${a})`;
}

/** One path for a whole connection list; the caller strokes it. */
function trace(ctx, pts, connections) {
  ctx.beginPath();

  for (const { start, end } of connections) {
    const a = pts[start], b = pts[end];
    if (!a || !b) continue;

    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
  }
}

/** An iris as a disc, from however many points the model gave us. */
function disc(ctx, pts, connections, colour, unit) {
  let x = 0, y = 0, n = 0;

  for (const { start } of connections) {
    const p = pts[start];
    if (!p) continue;
    x += p.x; y += p.y; n++;
  }

  if (!n) return;
  x /= n; y /= n;

  let r = 0;
  for (const { start } of connections) {
    const p = pts[start];
    if (p) r += Math.hypot(p.x - x, p.y - y);
  }
  r = Math.max(unit * 1.5, r / n);

  ctx.beginPath();
  ctx.arc(x, y, r, 0, Math.PI * 2);
  ctx.fillStyle = withAlpha(colour, 0.22);
  ctx.fill();
  ctx.strokeStyle = colour;
  ctx.lineWidth = Math.max(1, unit * 1.1);
  ctx.stroke();
}

/** Corner marks rather than a box: it says where the system is looking
 *  without drawing a dashed rectangle over the driver's chest. */
function brackets(ctx, [x1, y1, x2, y2], size) {
  ctx.beginPath();

  for (const [x, y, dx, dy] of [[x1, y1, 1, 1], [x2, y1, -1, 1],
                                [x1, y2, 1, -1], [x2, y2, -1, -1]]) {
    ctx.moveTo(x + dx * size, y);
    ctx.lineTo(x, y);
    ctx.lineTo(x, y + dy * size);
  }

  ctx.stroke();
}

export function draw(c, pts, state, shownHands, shownPhones, lastStrap, MAR_THRESHOLD) {
  const ctx = c.getContext("2d");

  ctx.clearRect(0, 0, c.width, c.height);
  if (overlay === "off") return;

  const style = getComputedStyle(document.documentElement);
  const ok = style.getPropertyValue("--ok").trim() || "#41c464";
  const bad = style.getPropertyValue("--bad").trim() || "#ff5f52";
  const warn = style.getPropertyValue("--warn").trim() || "#e5a234";
  const accent = style.getPropertyValue("--accent").trim() || "#4aa8ff";

  // Widths in proportion to the frame, so a 1280-wide laptop camera
  // and a 480-wide phone one look like the same design.
  const unit = Math.max(1, c.width / 640);
  const line = unit * 1.9;

  ctx.lineCap = "round";
  ctx.lineJoin = "round";

  if (pts) {
    const closed = state.blink?.eyes_closed;
    const yawning = (state.yawn?.mar ?? 0) > MAR_THRESHOLD;
    const offRoad = state.distraction && !state.distraction.eyes_on_road;

    // Colour carries state and nothing else. Everything calm is one
    // cool accent; red and amber appear only when a detector is
    // actually saying something, so a glance at the video answers "is
    // anything wrong" before any text is read. The old overlay was
    // bright green whenever things were FINE, which spent the loudest
    // colour available on the most common case.
    const eyes = closed ? bad : accent;
    const mouth = yawning ? warn : accent;
    const ray = offRoad ? bad : accent;

    if (overlay === "mesh" && MESH.length) {
      ctx.strokeStyle = withAlpha(accent, 0.11);
      ctx.lineWidth = Math.max(0.5, unit * 0.55);
      trace(ctx, pts, MESH);
      ctx.stroke();
    }

    ctx.strokeStyle = withAlpha(accent, 0.38);
    ctx.lineWidth = unit * 1.1;
    trace(ctx, pts, OVAL);
    ctx.stroke();

    ctx.strokeStyle = withAlpha(accent, 0.45);
    ctx.lineWidth = unit * 1.2;
    trace(ctx, pts, BROWS);
    ctx.stroke();

    // The glow is what stops a one-pixel contour disappearing into a
    // bright cabin, and it is cheap on four strokes.
    ctx.save();
    ctx.shadowColor = withAlpha(eyes, 0.8);
    ctx.shadowBlur = unit * 6;
    ctx.strokeStyle = eyes;
    ctx.lineWidth = line;
    trace(ctx, pts, EYES);
    ctx.stroke();

    ctx.shadowColor = withAlpha(mouth, 0.8);
    ctx.strokeStyle = mouth;
    trace(ctx, pts, LIPS);
    ctx.stroke();
    ctx.restore();

    for (const ring of IRISES) {
      if (ring.length) disc(ctx, pts, ring, eyes, unit);
    }

    // Where the head is pointing. Faded at the nose and solid at the
    // far end, so it reads as a direction rather than a whisker.
    const nose = pts[1];

    if (nose) {
      const len = c.width * 0.26;
      const tx = nose.x +
        Math.sin((state.head_pose?.yaw ?? 0) * Math.PI / 180) * len;
      const ty = nose.y +
        Math.sin((state.head_pose?.pitch ?? 0) * Math.PI / 180) * len;

      const fade = ctx.createLinearGradient(nose.x, nose.y, tx, ty);
      fade.addColorStop(0, withAlpha(ray, 0));
      fade.addColorStop(1, withAlpha(ray, 0.95));

      ctx.strokeStyle = fade;
      ctx.lineWidth = line * 1.15;
      ctx.beginPath();
      ctx.moveTo(nose.x, nose.y);
      ctx.lineTo(tx, ty);
      ctx.stroke();

      ctx.beginPath();
      ctx.arc(tx, ty, unit * 4.5, 0, Math.PI * 2);
      ctx.strokeStyle = ray;
      ctx.lineWidth = unit * 1.4;
      ctx.stroke();
    }
  }

  // The chest region whenever it was looked at, so an UNKNOWN seatbelt
  // verdict is visibly about framing rather than about the belt -- and
  // the strap itself when one was found.
  const observation = state.seatbelt?.observation;

  if (observation?.roi) {
    ctx.strokeStyle = observation.belt_found
      ? withAlpha(ok, 0.8) : withAlpha(warn, 0.5);
    ctx.lineWidth = unit * 1.6;
    brackets(ctx, observation.roi, unit * 14);
  }

  if (lastStrap) {
    ctx.save();
    ctx.shadowColor = withAlpha(ok, 0.9);
    ctx.shadowBlur = unit * 7;
    ctx.strokeStyle = ok;
    ctx.lineWidth = unit * 3;
    ctx.beginPath();
    for (const [x1, y1, x2, y2] of lastStrap) {
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
    }
    ctx.stroke();
    ctx.restore();
  }

  if (shownHands?.length) {
    const colour = state.wheel?.hands_off ? bad : ok;

    for (const hand of shownHands) {
      const scaled = hand.map(p => ({ x: p.x * c.width, y: p.y * c.height }));

      ctx.strokeStyle = withAlpha(colour, 0.75);
      ctx.lineWidth = unit * 1.8;
      trace(ctx, scaled, HAND_BONES);
      ctx.stroke();

      ctx.fillStyle = colour;
      for (const p of scaled) {
        ctx.beginPath();
        ctx.arc(p.x, p.y, unit * 2.2, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }

  if (shownPhones?.length) {
    const size = Math.max(11, Math.round(unit * 12));
    ctx.font = `600 ${size}px ${style.getPropertyValue("--sans").trim() ||
                                "sans-serif"}`;

    for (const box of shownPhones) {
      ctx.strokeStyle = bad;
      ctx.lineWidth = unit * 2;
      ctx.beginPath();
      ctx.roundRect?.(box.originX, box.originY, box.width, box.height,
                      unit * 4);
      if (!ctx.roundRect) {
        ctx.rect(box.originX, box.originY, box.width, box.height);
      }
      ctx.stroke();

      // A chip rather than bare text, which was unreadable against
      // anything pale.
      const label = "phone";
      const w = ctx.measureText(label).width + unit * 10;
      const h = size + unit * 7;
      const y = Math.max(0, box.originY - h - unit * 3);

      ctx.fillStyle = bad;
      ctx.beginPath();
      ctx.roundRect?.(box.originX, y, w, h, unit * 3);
      if (!ctx.roundRect) ctx.rect(box.originX, y, w, h);
      ctx.fill();

      ctx.fillStyle = "#12060a";
      ctx.textBaseline = "middle";
      ctx.fillText(label, box.originX + unit * 5, y + h / 2);
    }
  }
}

export function cycle() {
  overlay = overlay === "mesh" ? "outline"
          : overlay === "outline" ? "off" : "mesh";
  return overlay;
}

export function label() { return OVERLAY_LABEL[overlay]; }

export function mode() { return overlay; }

/*
 * Who is driving.
 *
 * The point of this is not a login. There is no server here to log in
 * to, and a password kept in localStorage is bypassed from the console
 * in one line, so anything calling itself authentication would be a
 * claim the code cannot support. What a driver monitor actually needs
 * from identity is narrower and real: somewhere to hang the things
 * that belong to a person rather than to a session.
 *
 * Today that is the eye threshold and the drive log.
 *
 * The threshold is worth keeping because eye shape varies enough
 * between people that one driver's open eye sits below another's blink
 * threshold, and learning it costs ten seconds of every single drive.
 * It is a property of the person, so it travels.
 *
 * The head pose offset is deliberately NOT kept, and that asymmetry is
 * the whole design. "Straight ahead" is not a property of the driver;
 * it is a property of where the phone is clamped today. Restoring
 * yesterday's offset onto a mount that has moved would teach the
 * system that sideways is forward and then score every glance at the
 * road as a distraction -- a stored calibration actively worse than no
 * calibration. It costs three seconds to measure properly, so it is
 * measured properly, every drive.
 *
 * Storage is the same deal as the drive log: this phone and nowhere
 * else, and every read assumes the data is missing, full or written by
 * some other version of this page.
 */

const KEY = "driver-monitor-drivers";

// Enough for a family car or a demo, few enough that the list stays a
// list rather than becoming a screen of its own.
const MAX_DRIVERS = 8;
const MAX_NAME = 40;

// ----------------------------------------------------------------------
// pure state, so the checks can exercise all of this without touching
// the storage a real driver's profiles live in
// ----------------------------------------------------------------------

export function empty() {
  return { drivers: [], currentId: null };
}

/**
 * Anything at all in, a usable state out.
 *
 * Nothing here throws and nothing here trusts: a half-written profile
 * from an older version of the page should cost that one profile, not
 * the start screen.
 */
export function validate(parsed) {
  if (!parsed || typeof parsed !== "object") return empty();

  const seen = new Set();
  const drivers = (Array.isArray(parsed.drivers) ? parsed.drivers : [])
    .filter(d => d && typeof d.id === "string" && d.id &&
                 typeof d.name === "string" && d.name.trim())
    .filter(d => !seen.has(d.id) && seen.add(d.id))
    .slice(0, MAX_DRIVERS)
    .map(d => ({
      id: d.id,
      name: d.name.trim().slice(0, MAX_NAME),
      created: Number.isFinite(d.created) ? d.created : Date.now(),
      ear: validEar(d.ear)
    }));

  const currentId = drivers.some(d => d.id === parsed.currentId)
    ? parsed.currentId
    : null;

  return { drivers, currentId };
}

/**
 * A threshold outside the range the calibrator itself would ever
 * produce is not this driver's eyes, it is corruption or a different
 * build's idea of the number, and it must not be loaded as fact.
 */
export function validEar(ear) {
  if (!ear || typeof ear !== "object") return null;
  if (!Number.isFinite(ear.threshold)) return null;
  if (ear.threshold <= 0 || ear.threshold >= 1) return null;

  return {
    threshold: ear.threshold,
    confidence: Number.isFinite(ear.confidence) ? ear.confidence : 1.0,
    at: Number.isFinite(ear.at) ? ear.at : null
  };
}

export function addTo(state, name) {
  const clean = String(name ?? "").trim().slice(0, MAX_NAME);
  if (!clean) return { state, driver: null, why: "A driver needs a name." };

  if (state.drivers.length >= MAX_DRIVERS) {
    return { state, driver: null, why: `Room for ${MAX_DRIVERS} drivers.` };
  }

  // Case-insensitive, because two profiles called Sam and sam are a
  // mistake every time, not a pair of people.
  if (state.drivers.some(d => d.name.toLowerCase() === clean.toLowerCase())) {
    return { state, driver: null, why: `${clean} is already here.` };
  }

  const driver = { id: newId(), name: clean, created: Date.now(), ear: null };

  return {
    state: { drivers: [...state.drivers, driver], currentId: driver.id },
    driver,
    why: null
  };
}

export function removeFrom(state, id) {
  const drivers = state.drivers.filter(d => d.id !== id);
  return {
    drivers,
    currentId: state.currentId === id ? null : state.currentId
  };
}

export function selectIn(state, id) {
  const exists = id !== null && state.drivers.some(d => d.id === id);
  return { drivers: state.drivers, currentId: exists ? id : null };
}

/** Writes a calibration onto one driver, leaving everything else as is. */
export function calibrationIn(state, id, ear) {
  const valid = ear === null ? null : validEar(ear);

  return {
    drivers: state.drivers.map(d => d.id === id ? { ...d, ear: valid } : d),
    currentId: state.currentId
  };
}

export function currentIn(state) {
  return state.drivers.find(d => d.id === state.currentId) ?? null;
}

// ----------------------------------------------------------------------
// the stored copy
// ----------------------------------------------------------------------

export function load() {
  try {
    const raw = localStorage.getItem(KEY);
    return raw ? validate(JSON.parse(raw)) : empty();
  } catch (e) {
    return empty();
  }
}

export function save(state) {
  try {
    localStorage.setItem(KEY, JSON.stringify(state));
    return true;
  } catch (e) {
    // No storage, or no room. The profile still works for this drive;
    // it simply will not be there tomorrow.
    return false;
  }
}

function newId() {
  try {
    if (crypto?.randomUUID) return crypto.randomUUID();
  } catch (e) { /* fall through */ }

  return `d${Date.now().toString(36)}${Math.random().toString(36).slice(2, 8)}`;
}

export const LIMITS = { MAX_DRIVERS, MAX_NAME };

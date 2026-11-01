/*
 * The drive log.
 *
 * Kept in localStorage, which means it is kept on this phone and
 * nowhere else. The desktop writes its sessions to SQLite for the
 * analytics dashboard; there is no server here to write to, and adding
 * one would mean uploading a record of how tired someone was at work,
 * which is the kind of data this project has deliberately kept local
 * from the start.
 *
 * Every read is defensive. Storage can be unavailable (private
 * browsing, blocked site data), full, or holding something another
 * version of this page wrote, and a report screen that throws is worse
 * than one that says there is no history.
 */

const KEY = "driver-monitor-drives";

// Enough to show a trend, few enough that the quota is never the thing
// that breaks. Per driver, because a shared phone where one person
// drives daily and another weekly would otherwise quietly delete the
// second person's whole history to make room -- and a total as well,
// so eight profiles cannot multiply into a quota problem.
const KEEP_PER_DRIVER = 20;
const KEEP_TOTAL = 60;

/**
 * Newest first, each driver capped, then the whole list capped.
 *
 * Pure, so the checks can run it on made-up drives without touching
 * the log on this device.
 */
export function trim(drives) {
  const counts = new Map();
  const kept = [];

  for (const drive of [...drives].sort((a, b) => b.at - a.at)) {
    const key = ownerOf(drive) ?? "";

    if ((counts.get(key) ?? 0) >= KEEP_PER_DRIVER) continue;

    counts.set(key, (counts.get(key) ?? 0) + 1);
    kept.push(drive);

    if (kept.length >= KEEP_TOTAL) break;
  }

  return kept;
}

/**
 * Drives belonging to one driver, or -- for null -- the ones belonging
 * to nobody. That second case is not a special case bolted on: it is
 * where every drive recorded before profiles existed lives, and where
 * a guest drive goes, so the log from before this feature is still
 * readable instead of being orphaned by it.
 */
export function belongTo(drives, driverId) {
  const want = driverId ?? null;
  return drives.filter(drive => ownerOf(drive) === want);
}

export function ownerOf(drive) {
  return typeof drive?.driverId === "string" && drive.driverId
    ? drive.driverId
    : null;
}

export function load() {
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return [];

    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];

    return trim(parsed.filter(
      drive => drive && typeof drive.at === "number"));
  } catch (e) {
    return [];
  }
}

export function save(drive) {
  try {
    const drives = trim([drive, ...load()]);
    localStorage.setItem(KEY, JSON.stringify(drives));
    return true;
  } catch (e) {
    // Out of quota, or no storage at all. The drive just finished is
    // still on screen; only the history is lost, so there is nothing
    // to tell the driver about mid-report.
    return false;
  }
}

export function clear() {
  try {
    localStorage.removeItem(KEY);
    return true;
  } catch (e) {
    return false;
  }
}

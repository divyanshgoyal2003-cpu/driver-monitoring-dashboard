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
// that breaks.
const KEEP = 20;

export function load() {
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return [];

    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];

    return parsed
      .filter(drive => drive && typeof drive.at === "number")
      .sort((a, b) => b.at - a.at)
      .slice(0, KEEP);
  } catch (e) {
    return [];
  }
}

export function save(drive) {
  try {
    const drives = [drive, ...load()].slice(0, KEEP);
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

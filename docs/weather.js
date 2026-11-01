/*
 * Weather, ported from services/weather.py and the WMO table in
 * app/config.py.
 *
 * Why the score cares at all: this is not a claim that rain makes a
 * driver sleepier. It is about what a lapse costs. A wet road lengthens
 * the stopping distance and fog shortens the warning, so the same
 * two-second glance away leaves less room to recover from -- and so
 * does the same microsleep. The mechanism is identical for both
 * families, which is why there is one table and not two.
 *
 * Where the location comes from is the one thing this does differently
 * from the desktop. There, coordinates are typed into a config file and
 * nothing happens until they are. Here the browser asks permission once
 * and the driver answers it, which is the same consent made explicit --
 * and, unlike an IP lookup, no third party is told anything in order to
 * guess where the phone is. Coordinates are rounded before they are
 * sent, because a forecast is a city-scale thing and the extra decimals
 * would only be precision given away for nothing.
 *
 * `status` exists because this used to be silent about its own state.
 * It worked -- it fetched, it interpreted, it returned "clear" -- and
 * "clear x1.00" read exactly like a forecast that had never arrived.
 * A reading of 1.00 from a clear sky and a 1.00 from ignorance weigh
 * the same and mean completely different things, so the panel has to
 * be able to tell them apart.
 */

export const WEATHER_MULTIPLIERS = {
  clear: 1.00,
  cloudy: 1.00,
  rain: 1.15,
  heavy_rain: 1.25,
  fog: 1.25,
  snow: 1.30,
  ice: 1.35
};

// WMO codes as Open-Meteo reports them. A code that is not listed is
// treated as NO READING rather than as "clear": a forecast the system
// cannot interpret must not silently remove the amplification.
const WMO = {
  0: "clear", 1: "clear",
  2: "cloudy", 3: "cloudy",
  45: "fog", 48: "fog",
  51: "rain", 53: "rain", 55: "rain",
  56: "ice", 57: "ice",
  61: "rain", 63: "rain", 65: "heavy_rain",
  66: "ice", 67: "ice",
  71: "snow", 73: "snow", 75: "snow", 77: "snow",
  80: "rain", 81: "rain", 82: "heavy_rain",
  85: "snow", 86: "snow",
  95: "heavy_rain", 96: "heavy_rain", 99: "heavy_rain"
};

const URL = "https://api.open-meteo.com/v1/forecast";
const REFRESH = 900.0;    // the API's `current` block updates this often
const TIMEOUT = 8000;

const ICE_TEMPERATURE = 0.5;
const FOG_VISIBILITY = 1000.0;

// Where the last known coordinates are kept, so the second drive needs
// no prompt and no input at all.
const STORED = "driver-monitor-location";

// Ordered worst-last, so "pick the worse of two readings" is a
// comparison of positions in this list.
const SEVERITY_ORDER =
  ["clear", "cloudy", "rain", "heavy_rain", "fog", "snow", "ice"];

function worse(a, b) {
  if (a === null) return b;
  if (b === null) return a;
  return SEVERITY_ORDER.indexOf(b) > SEVERITY_ORDER.indexOf(a) ? b : a;
}

/**
 * Two refinements on top of the reported code, because the code alone
 * can understate what the road is like.
 *
 * Liquid rain at or below freezing is ice on the tarmac whatever the
 * forecast calls it, and visibility this low is fog-grade however clear
 * the sky is said to be. Both only ever pick the WORSE of the two
 * readings. Snow is exempt from the first: it lands already frozen and
 * grips better than a freezing-rain glaze.
 */
export function interpret(code, temperature, visibility) {
  let condition = WMO[code] ?? null;

  if (condition === null) return null;

  const liquid = condition === "rain" || condition === "heavy_rain";

  if (liquid && typeof temperature === "number" &&
      temperature <= ICE_TEMPERATURE) {
    condition = worse(condition, "ice");
  }

  if (typeof visibility === "number" && visibility <= FOG_VISIBILITY) {
    condition = worse(condition, "fog");
  }

  return condition;
}

export class WeatherService {
  constructor() {
    this.condition = null;
    this.temperature = null;
    this.visibility = null;
    this.fetchedAt = null;
    this.coordinates = recall();
    this.error = null;
    this.pending = false;

    // idle | locating | fetching | ready | failed
    this.status = this.coordinates ? "idle" : "idle";
  }

  /** (condition, source). The source travels with the reading because
   *  a 1.00 from a clear sky and a 1.00 from ignorance are not the
   *  same evidence, and the panel says which it is. */
  provenance() {
    if (this.condition === null) return [null, "default"];
    return [this.condition, "forecast"];
  }

  get located() {
    return this.coordinates !== null;
  }

  /**
   * Asks the browser for a location. The prompt is the driver's
   * consent; nothing is sent anywhere until they give it, and once
   * they have, the coordinates are remembered so no later drive asks
   * again.
   */
  async locate() {
    if (!navigator.geolocation) {
      this.status = "failed";
      this.error = "This browser has no location service.";
      return false;
    }

    this.status = "locating";
    this.error = null;

    try {
      const position = await new Promise((resolve, reject) =>
        navigator.geolocation.getCurrentPosition(resolve, reject, {
          timeout: 10000, maximumAge: 600000, enableHighAccuracy: false
        })
      );

      // Two decimals is about a kilometre, which is finer than a
      // forecast resolves anyway.
      this.coordinates = {
        latitude: Math.round(position.coords.latitude * 100) / 100,
        longitude: Math.round(position.coords.longitude * 100) / 100
      };

      remember(this.coordinates);
      this.status = "idle";
      return true;
    } catch (e) {
      this.status = "failed";
      this.error = e && e.code === 1
        ? "Location permission was declined."
        : "Could not get a location.";
      return false;
    }
  }

  /** Fetches at most every REFRESH seconds; returns true if it tried. */
  async refresh(t) {
    if (!this.coordinates || this.pending) return false;
    if (this.fetchedAt !== null && t - this.fetchedAt < REFRESH) return false;

    this.pending = true;
    this.status = "fetching";

    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), TIMEOUT);

    try {
      const query = new URLSearchParams({
        latitude: this.coordinates.latitude,
        longitude: this.coordinates.longitude,
        current: "weather_code,temperature_2m,visibility"
      });

      const response = await fetch(`${URL}?${query}`, { signal: controller.signal });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);

      const body = await response.json();
      const current = body.current ?? {};

      this.condition = interpret(
        current.weather_code,
        current.temperature_2m,
        current.visibility
      );

      this.temperature = current.temperature_2m ?? null;
      this.visibility = current.visibility ?? null;
      this.fetchedAt = t;

      if (this.condition === null) {
        this.status = "failed";
        this.error = `Forecast code ${current.weather_code} means nothing here.`;
      } else {
        this.status = "ready";
        this.error = null;
      }
    } catch (e) {
      // A forecast that did not arrive falls back to no reading, which
      // weighs 1.0. A system that cannot see the weather must not
      // invent an amplification.
      this.condition = null;
      this.status = "failed";
      this.error = e.name === "AbortError"
        ? "The forecast timed out."
        : "The forecast could not be fetched.";
      this.fetchedAt = t;
    } finally {
      clearTimeout(timer);
      this.pending = false;
    }

    return true;
  }

  /**
   * Everything needed for one drive, asked for once and without the
   * driver having to press anything. Returns true if there is a
   * reading at the end of it.
   */
  async begin(t) {
    if (!this.coordinates && !(await this.locate())) return false;
    await this.refresh(t);
    return this.condition !== null;
  }

  forget() {
    this.coordinates = null;
    this.condition = null;
    this.fetchedAt = null;
    this.status = "idle";
    this.error = null;
    try { localStorage.removeItem(STORED); } catch (e) { /* no storage */ }
  }
}

// Storage can be unavailable, full, or holding something another
// version of this page wrote, and a forecast is not worth throwing on.
function remember(coordinates) {
  try {
    localStorage.setItem(STORED, JSON.stringify(coordinates));
  } catch (e) { /* the prompt will simply come again next time */ }
}

function recall() {
  try {
    const raw = localStorage.getItem(STORED);
    if (!raw) return null;

    const parsed = JSON.parse(raw);
    const ok = parsed &&
      Number.isFinite(parsed.latitude) && Number.isFinite(parsed.longitude) &&
      Math.abs(parsed.latitude) <= 90 && Math.abs(parsed.longitude) <= 180;

    return ok ? { latitude: parsed.latitude, longitude: parsed.longitude } : null;
  } catch (e) {
    return null;
  }
}

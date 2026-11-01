/*
 * Offline support.
 *
 * A driver monitor that stops working in a tunnel, a car park or a
 * patch of no signal would be worse than useless, so everything the
 * app needs is cached on first run: the page, the code, the fonts and
 * the 3.6 MB face model. After that it runs with the network off.
 *
 * Bump CACHE whenever the shell changes, or browsers will keep serving
 * the old one.
 */

const CACHE = "driver-monitor-v2";

// The two optional models are NOT in here. They are opt-in, they are
// 15 MB between them, and pre-caching them would hand that bill to
// every driver including the ones who never tick either box. They are
// picked up by the fetch handler below the first time they are
// actually used.
const SHELL = [
  "./",
  "./index.html",
  "./app.css",
  "./app.js",
  "./detectors.js",
  "./vision.js",
  "./weather.js",
  "./history.js",
  "./manifest.webmanifest",
  "./icon.svg",
  "./icon-maskable.svg",
  "./face_landmarker.task"
];

self.addEventListener("install", event => {
  event.waitUntil(
    caches.open(CACHE)
      // Individually, so one failed file does not throw away the rest.
      .then(cache => Promise.allSettled(SHELL.map(url => cache.add(url))))
      .then(() => self.skipWaiting())
  );
});

self.addEventListener("activate", event => {
  event.waitUntil(
    caches.keys()
      .then(keys => Promise.all(
        keys.filter(key => key !== CACHE).map(key => caches.delete(key))
      ))
      .then(() => self.clients.claim())
  );
});

self.addEventListener("fetch", event => {
  const request = event.request;

  if (request.method !== "GET") return;

  // The forecast is the one thing here that is supposed to change, so
  // it is never served from the cache. Offline it simply fails, and
  // the weather service falls back to no reading, which weighs 1.0.
  if (request.url.includes("api.open-meteo.com")) return;

  // Cache first: these files never change without a new CACHE name, and
  // the model is far too big to re-fetch on every start.
  event.respondWith(
    caches.match(request).then(hit => {
      if (hit) return hit;

      return fetch(request).then(response => {
        // MediaPipe's WASM comes from a CDN on first run; keep it so the
        // second run needs no network at all.
        if (response.ok &&
            (request.url.startsWith(self.location.origin) ||
             request.url.includes("tasks-vision") ||
             request.url.includes("fonts.g"))) {
          const copy = response.clone();
          caches.open(CACHE).then(cache => cache.put(request, copy));
        }
        return response;
      }).catch(() => hit);
    })
  );
});

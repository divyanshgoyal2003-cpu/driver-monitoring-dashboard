/*
 * Pulls the two things docs/ still fetches over the network into
 * vendor/, so the desktop build does not need a connection.
 *
 * The models are already local; what is not is the MediaPipe runtime
 * (the JS bundle and its wasm) and the webfont. In a browser that is
 * fine -- the service worker caches them on first run. The desktop
 * build has no service worker, and a review room is exactly where the
 * wifi is worst, so they ship inside the executable instead.
 *
 * docs/ is NOT edited to point here. It still asks for the CDN URLs
 * it asks for on GitHub Pages, and main.js answers those requests
 * from this folder. One copy of the web app, no desktop fork of it.
 *
 *   node fetch-vendor.js
 */

const fs = require("node:fs");
const path = require("node:path");

const TASKS = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";
const VENDOR = path.join(__dirname, "vendor");

/*
 * FilesetResolver asks for whichever of these the device can run, and
 * which one that is depends on the browser's SIMD support -- so both
 * builds have to be here or the fallback path is a 404 at the worst
 * possible moment.
 */
const TASK_FILES = [
  "vision_bundle.mjs",
  "wasm/vision_wasm_internal.js",
  "wasm/vision_wasm_internal.wasm",
  "wasm/vision_wasm_nosimd_internal.js",
  "wasm/vision_wasm_nosimd_internal.wasm"
];

const FONT_CSS =
  "https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600" +
  "&family=IBM+Plex+Sans:wght@400;500;600&display=swap";

async function get(url) {
  const res = await fetch(url, {
    // Google Fonts serves a different stylesheet to browsers that
    // cannot do woff2. Asking as a browser gets the small files.
    headers: { "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) " +
                             "AppleWebKit/537.36 (KHTML, like Gecko) " +
                             "Chrome/120.0.0.0 Safari/537.36" }
  });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText} for ${url}`);
  return Buffer.from(await res.arrayBuffer());
}

function put(rel, buf) {
  const file = path.join(VENDOR, rel);
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(file, buf);
  console.log(String(buf.length).padStart(9), rel);
}

async function main() {
  for (const rel of TASK_FILES) {
    put(path.join("tasks-vision", rel), await get(`${TASKS}/${rel}`));
  }

  /*
   * The stylesheet names its font files by absolute URL. Each one is
   * fetched and the URL rewritten to a relative path, because the
   * whole point is that nothing here reaches fonts.gstatic.com at
   * runtime.
   */
  let css = (await get(FONT_CSS)).toString("utf8");
  const urls = [...new Set(css.match(/https:\/\/fonts\.gstatic\.com\/[^)]+/g) || [])];

  for (const url of urls) {
    const name = url.split("/").pop().split("?")[0];
    put(path.join("fonts", name), await get(url));
    css = css.split(url).join("./fonts/" + name);
  }

  put("font.css", Buffer.from(css, "utf8"));
  console.log(`\n${TASK_FILES.length} runtime files, ${urls.length} fonts`);
}

main().catch(e => { console.error("failed:", e.message); process.exit(1); });

/*
 * `npm run verify`, or `DriverMonitor.exe --self-test`.
 *
 * A desktop build can fail in ways the browser build cannot, and none
 * of them are visible in a screenshot: a module that will not import
 * because the loopback server got a header wrong, a model that 404s
 * because it was left out of the package, a camera the permission
 * handler quietly refused, a CDN rewrite that did not fire and left
 * the app needing the network it was built to do without.
 *
 * So the window checks itself and the process exits with the verdict.
 */

const { app, BrowserWindow } = require("electron");

const CHECKS = `(async () => {
  const out = { };

  // 1. The app's own suite, running on the same files in the window
  //    they ship in. textContent rather than innerText, because the
  //    window is never shown and an unlaid-out document has no
  //    rendered text to read.
  //    No regex: this whole block is a template literal on the way
  //    here, and a template literal eats the backslash out of \d.
  const tally = document.getElementById("tally");
  // "131 passed, 0 failed" -- note that splitting on spaces leaves the
  // comma attached to "passed,", so the words are not what is read.
  const said = (tally ? tally.textContent : "").split(" ");
  const run = Number(said[0]), bad = Number(said[2]);
  out.suite = said.length >= 4 && Number.isFinite(run) && Number.isFinite(bad)
    ? { passed: run, failed: bad } : { raw: tally ? tally.textContent : "(none)" };

  // 2. The runtime and a model, loaded the way the app loads them.
  //    This is the one that proves the package is complete.
  try {
    const V = await import(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs");
    const files = await V.FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm");
    const face = await V.FaceLandmarker.createFromOptions(files, {
      baseOptions: { modelAssetPath: "./face_landmarker.task", delegate: "GPU" },
      runningMode: "VIDEO", numFaces: 1
    });
    out.model = face ? "face_landmarker loaded" : "no landmarker";
    face.close();
  } catch (e) { out.model = "FAILED: " + e.message; }

  // 3. The camera. Opened and released -- if the permission handler
  //    is wrong this is where it shows, not on the reviewer's desk.
  try {
    const s = await navigator.mediaDevices.getUserMedia({ video: true });
    const t = s.getVideoTracks()[0].getSettings();
    out.camera = t.width + "x" + t.height + " @ " + Math.round(t.frameRate || 0) + " fps";
    s.getTracks().forEach(x => x.stop());
  } catch (e) { out.camera = "FAILED: " + e.message; }

  // 4. The location Windows knows, which Chromium cannot ask for.
  //    First call can take a few seconds while the service warms up.
  try {
    const p = await new Promise((res, rej) =>
      navigator.geolocation.getCurrentPosition(res, rej, { timeout: 20000 }));
    out.location = p.coords.latitude.toFixed(2) + ", " +
                   p.coords.longitude.toFixed(2) +
                   " (" + Math.round(p.coords.accuracy) + " m)";
  } catch (e) { out.location = "FAILED: " + (e.message || e.code); }

  // 5. Fonts, which are the quiet one: a missing webfont does not
  //    throw, it just makes the app look like a different app.
  try {
    await document.fonts.ready;
    out.font = document.fonts.check("12px 'IBM Plex Mono'") ? "loaded" : "NOT loaded";
  } catch (e) { out.font = "FAILED: " + e.message; }

  return out;
})()`;

async function run(serverUrl, prepare) {
  const win = new BrowserWindow({ show: false, width: 1200, height: 900 });
  // The real window gets a working geolocation from the shell, so the
  // window being checked has to get it too or the check means nothing.
  if (prepare) prepare(win);
  const say = [];
  let bad = false;

  const note = (name, value, ok) => {
    say.push(`  ${ok ? "ok  " : "FAIL"}  ${name.padEnd(22)} ${value}`);
    if (!ok) bad = true;
  };

  // A page that throws on load reports "no result" and says nothing
  // about why. Its own errors are worth more than that.
  win.webContents.on("console-message", (e, level, message) => {
    const first = String(message).split(String.fromCharCode(10))[0];
    if (level >= 2 && !first.includes("Electron Security Warning")) {
      say.push("  note  " + first.slice(0, 120));
    }
  });

  try {
    await win.loadURL(serverUrl + "/test.html");
    // The suite is synchronous on load; the imports above are not.
    const r = await win.webContents.executeJavaScript(CHECKS, true);

    const suite = r.suite || {};
    note("self-test suite",
         suite.passed === undefined
           ? `could not read the tally: ${suite.raw ?? "no result"}`
           : `${suite.passed} passed, ${suite.failed} failed`,
         suite.failed === 0 && suite.passed > 0);
    note("mediapipe + model", r.model, !String(r.model).startsWith("FAILED"));
    note("camera", r.camera, !String(r.camera).startsWith("FAILED"));
    note("location (windows)", r.location, !String(r.location).startsWith("FAILED"));
    note("webfont", r.font, r.font === "loaded");

    const n = global.__rewrites || 0;
    note("cdn served locally", `${n} request${n === 1 ? "" : "s"} rewritten`, n > 0);
  } catch (e) {
    note("ran at all", String(e), false);
  }

  console.log("\n" + say.join("\n"));
  console.log(bad ? "\nSELF-TEST FAILED\n" : "\nSELF-TEST PASSED\n");

  win.destroy();
  app.exit(bad ? 1 : 0);
}

module.exports = { run };

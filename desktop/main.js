/*
 * The desktop build of Driver Monitor.
 *
 * This window does not contain a port of the web app. It contains the
 * web app -- the same docs/ folder, served to the window over
 * loopback, so there is exactly one copy of the detectors, the scoring
 * and the interface, and no way for a desktop version to drift away
 * from the one being reviewed. Everything in here is window dressing
 * around that: a server, a camera permission, a menu, and the two
 * things the platform does not supply on its own -- the MediaPipe
 * runtime, which is on a CDN, and a location, which Chromium cannot
 * reach without a Google API key.
 *
 * What the desktop build gets that the browser does not: no address
 * bar, no tab to lose, no service worker (so no stale cache -- the
 * files are read off the disk every launch), and nothing it needs
 * from the network. The models were already local; the runtime and
 * the webfont are now too.
 */

const { app, BrowserWindow, Menu, shell, dialog } = require("electron");
const path = require("node:path");
const fs = require("node:fs");
const { serve } = require("./serve.js");
const { where } = require("./location.js");

/*
 * Packaged, docs/ is dropped beside the executable's resources.
 * Running from source it is the sibling folder in the repository --
 * which means `npm start` is driving the very files being edited, and
 * a reload shows an edit immediately.
 */
const DOCS = app.isPackaged
  ? path.join(process.resourcesPath, "docs")
  : path.join(__dirname, "..", "docs");

// Written by fetch-vendor.js, which npm install runs. Inside the
// packaged app this path is in the asar, which fs reads straight
// through.
const VENDOR = path.join(__dirname, "vendor");

/*
 * The CDN addresses docs/ asks for, and where this build answers them
 * from instead. fonts.gstatic.com -- where the font files themselves
 * live -- needs no entry: the stylesheet was rewritten on download to
 * name them by relative path, so they resolve under /__vendor/ on
 * their own.
 */
const OFFLINE = [
  ["https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/", "tasks-vision/"],
  ["https://fonts.googleapis.com/css2", "font.css"]
];

let server = null;
let win = null;

/*
 * Two copies would both bind a port and both fight over the same
 * localStorage, so the second launch hands its turn to the first and
 * exits.
 */
if (!app.requestSingleInstanceLock()) {
  app.quit();
} else {
  app.on("second-instance", () => {
    if (!win) return;
    if (win.isMinimized()) win.restore();
    win.focus();
  });

  app.whenReady().then(startUp);
}

async function startUp() {
  if (!fs.existsSync(path.join(DOCS, "index.html"))) {
    dialog.showErrorBox(
      "Driver Monitor",
      "The application files are missing.\n\nExpected them at:\n" + DOCS
    );
    app.quit();
    return;
  }

  try {
    server = await serve([
      { prefix: "/", root: DOCS },
      { prefix: "/__vendor/", root: VENDOR, cors: true }
    ], { "/__location": where });
  } catch (e) {
    dialog.showErrorBox(
      "Driver Monitor",
      "Could not start the local server.\n\n" + String(e)
    );
    app.quit();
    return;
  }

  grantTheCamera();
  serveTheRuntimeLocally();

  // `--self-test` opens no window the reviewer can see: it checks the
  // build and exits with the verdict. See selftest.js for why a
  // desktop build needs one.
  if (process.argv.includes("--self-test")) {
    require("./selftest.js").run(server.url, lendItAWorkingGeolocation);
    return;
  }

  Menu.setApplicationMenu(buildMenu());
  openWindow();
}

/*
 * The camera is the entire application, and this window only ever
 * loads files shipped inside it -- there is no page here that could
 * ask for a camera the driver did not come for. So the request is
 * granted without a second prompt on top of the one Windows already
 * enforces in its own privacy settings.
 *
 * Everything else is refused by name rather than by omission, so that
 * a later Electron adding a new permission does not quietly inherit a
 * yes.
 */
function grantTheCamera() {
  const { session } = require("electron");
  const allowed = new Set(["media", "audioCapture", "videoCapture"]);

  session.defaultSession.setPermissionRequestHandler(
    (contents, permission, callback) => callback(allowed.has(permission))
  );

  // navigator.permissions.query() goes through a separate handler, and
  // a camera that reads as "denied" there can stop a page asking at all.
  session.defaultSession.setPermissionCheckHandler(
    (contents, permission) => allowed.has(permission)
  );
}

/*
 * docs/ loads the MediaPipe runtime and the webfont from a CDN,
 * because on GitHub Pages that is the right answer -- the service
 * worker caches them on first run and the app works in a tunnel after
 * that. This build has no service worker and may well be opened on a
 * machine with no connection at all, so those requests are answered
 * from vendor/ instead.
 *
 * The rewrite happens here rather than in docs/, so the web app keeps
 * exactly the source it has in the browser and there is no desktop
 * copy of it to maintain.
 *
 * If vendor/ is missing -- somebody ran electron directly without an
 * npm install -- nothing is rewritten and the requests go to the CDN
 * as they always did. A missing download should cost the offline
 * guarantee, not the application.
 */
function serveTheRuntimeLocally() {
  if (!fs.existsSync(path.join(VENDOR, "tasks-vision", "vision_bundle.mjs"))) {
    console.warn("vendor/ is missing; the runtime will load from the CDN.");
    return;
  }

  const { session } = require("electron");
  const filter = { urls: OFFLINE.map(([prefix]) => prefix + "*") };

  session.defaultSession.webRequest.onBeforeRequest(filter, (details, callback) => {
    for (const [prefix, local] of OFFLINE) {
      if (!details.url.startsWith(prefix)) continue;

      // Whatever follows the prefix is the path inside the package;
      // for the stylesheet the query string is dropped, since the one
      // that was downloaded already asked for those families.
      const rest = local.endsWith("/") ? details.url.slice(prefix.length) : "";
      global.__rewrites = (global.__rewrites || 0) + 1;
      callback({ redirectURL: `${server.url}/__vendor/${local}${rest}` });
      return;
    }
    callback({});
  });
}

function openWindow() {
  win = new BrowserWindow({
    width: 1200,
    height: 900,
    minWidth: 360,
    minHeight: 560,
    // Matches the app's own background, so a slow first paint shows the
    // right colour instead of a white flash.
    backgroundColor: "#0a0e13",
    autoHideMenuBar: true,
    title: "Driver Monitor",
    icon: iconPath(),
    webPreferences: {
      // Nothing in docs/ asks for a native API, so it does not get
      // one. The page runs with the same powers it has in a browser.
      nodeIntegration: false,
      contextIsolation: true,
      sandbox: true
    }
  });

  lendItAWorkingGeolocation(win);
  win.loadURL(server.url + "/index.html");

  // A link in the interface is a link to the web, and belongs in the
  // browser rather than replacing the monitor in this window.
  win.webContents.setWindowOpenHandler(({ url }) => {
    if (/^https?:/.test(url)) shell.openExternal(url);
    return { action: "deny" };
  });

  win.webContents.on("will-navigate", (event, url) => {
    if (!url.startsWith(server.url)) {
      event.preventDefault();
      if (/^https?:/.test(url)) shell.openExternal(url);
    }
  });

  win.on("closed", () => { win = null; });
}

/*
 * The one browser API that does not work in a packaged Electron app.
 * Chromium asks a Google service that wants an API key; there is none,
 * so getCurrentPosition fails with POSITION_UNAVAILABLE and the
 * forecast never arrives. location.js gets the same answer out of
 * Windows instead.
 *
 * Only getCurrentPosition is replaced, and only on the instance, so
 * the rest of the API is whatever Chromium provides. If Windows has no
 * answer either -- the service is off, or the user has denied location
 * to desktop apps -- the call is handed to the original, which fails
 * exactly as it would have, and the app says "Could not get a
 * location" and carries on with the weather dropdown.
 *
 * This is the same bargain as the CDN rewrite: the shell supplies what
 * the platform did not, and docs/ is left alone.
 */
function lendItAWorkingGeolocation(target) {
  const PATCH = `(() => {
    const geo = navigator.geolocation;
    if (!geo) return "no geolocation to patch";
    const original = geo.getCurrentPosition.bind(geo);

    geo.getCurrentPosition = function (ok, fail, options) {
      fetch("/__location")
        .then(r => r.json())
        .then(d => {
          if (!d || typeof d.latitude !== "number") throw new Error("unknown");
          ok({
            coords: {
              latitude: d.latitude, longitude: d.longitude,
              accuracy: d.accuracy, altitude: null, altitudeAccuracy: null,
              heading: null, speed: null
            },
            timestamp: Date.now()
          });
        })
        .catch(() => original(ok, fail, options));
    };
    return "patched";
  })()`;

  // dom-ready, not did-finish-load: the app only asks for a location
  // when the driver clicks Locate, but patching before the page has
  // finished loading leaves no window where the old one is reachable.
  target.webContents.on("dom-ready", () => {
    target.webContents.executeJavaScript(PATCH, true).catch(() => {});
  });
}

function iconPath() {
  const ico = path.join(__dirname, "build", "icon.ico");
  return fs.existsSync(ico) ? ico : undefined;
}

/*
 * The menu is hidden behind Alt. It exists because a demo machine
 * eventually needs a reload, a zoom or a console, and because removing
 * the menu altogether takes the keyboard shortcuts with it.
 */
function buildMenu() {
  return Menu.buildFromTemplate([
    {
      label: "App",
      submenu: [
        { role: "reload" },
        { role: "forceReload" },
        { type: "separator" },
        { role: "resetZoom" },
        { role: "zoomIn" },
        { role: "zoomOut" },
        { type: "separator" },
        { role: "togglefullscreen" },
        { role: "toggleDevTools" },
        { type: "separator" },
        {
          label: "Self-test",
          click: () => {
            if (win) win.loadURL(server.url + "/test.html");
          }
        },
        {
          label: "Back to the monitor",
          click: () => {
            if (win) win.loadURL(server.url + "/index.html");
          }
        },
        { type: "separator" },
        { role: "quit" }
      ]
    }
  ]);
}

app.on("window-all-closed", async () => {
  if (server) await server.close();
  app.quit();
});

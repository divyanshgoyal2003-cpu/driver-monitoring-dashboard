/*
 * A static file server bound to loopback, so the desktop app can load
 * the web app unchanged.
 *
 * The obvious thing is to point the window at file:// and skip the
 * server. That does not work here, and the reasons are worth writing
 * down because they are the whole justification for this file:
 *
 *   - The app is ES modules. A module loaded over file:// has a null
 *     origin, so every `import` in it fails the CORS check.
 *   - getUserMedia needs a secure context. http://127.0.0.1 counts as
 *     one; file:// does not.
 *   - The models are fetched, not imported, and fetch() refuses
 *     file:// outright.
 *
 * So the window talks HTTP to a server inside the same process. Only
 * 127.0.0.1 is bound -- nothing on the network can reach it -- and the
 * port is whatever the OS hands out, so two copies cannot collide.
 */

const http = require("node:http");
const fs = require("node:fs");
const path = require("node:path");

const TYPES = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".webmanifest": "application/manifest+json; charset=utf-8",
  ".svg": "image/svg+xml",
  ".png": "image/png",
  ".ico": "image/x-icon",
  ".woff2": "font/woff2",
  // Not octet-stream. WebAssembly.compileStreaming refuses anything
  // else and silently falls back to buffering the whole module first,
  // which costs a second of startup for nothing.
  ".wasm": "application/wasm",
  // The three MediaPipe models. Nothing parses these but MediaPipe,
  // and it only cares that the bytes arrive intact.
  ".task": "application/octet-stream",
  ".tflite": "application/octet-stream"
};

/*
 * A request for "/../../../etc/passwd" has to land nowhere. Resolving
 * the path and then checking it is still inside the root is the only
 * check that survives the encodings -- stripping ".." by hand misses
 * "%2e%2e" and the backslash spellings Windows accepts.
 */
function resolveInside(root, urlPath) {
  let decoded;
  try {
    decoded = decodeURIComponent(urlPath.split("?")[0].split("#")[0]);
  } catch (e) {
    return null;
  }

  if (decoded.endsWith("/")) decoded += "index.html";
  if (decoded === "") decoded = "/index.html";

  const full = path.resolve(root, "." + decoded);
  const inside = full === root || full.startsWith(root + path.sep);
  return inside ? full : null;
}

/*
 * Mounts are tried longest prefix first, so a specific one wins over
 * "/". Each is { prefix, root, cors }.
 *
 * The second mount is the vendored MediaPipe runtime and webfont. It
 * is reached only because main.js rewrites the CDN addresses docs/
 * asks for -- docs/ itself has no idea this exists, which is the point:
 * the files served here are the same files GitHub Pages serves.
 *
 * It answers with Access-Control-Allow-Origin, because a request that
 * STARTED cross-origin stays CORS-checked after it is redirected, even
 * once it has landed back on this origin. Without the header the module
 * import fails. Nothing off this machine can reach the port.
 */
function serve(mounts, routes = {}) {
  const mounted = (Array.isArray(mounts) ? mounts : [{ prefix: "/", root: mounts }])
    .map(m => ({ ...m, base: path.resolve(m.root) }))
    .sort((a, b) => b.prefix.length - a.prefix.length);

  const server = http.createServer((req, res) => {
    if (req.method !== "GET" && req.method !== "HEAD") {
      res.writeHead(405).end();
      return;
    }

    const url = req.url || "/";

    // A route answers instead of a file. There is one: the location
    // Windows knows and Chromium cannot ask for. See location.js.
    const route = routes[url.split("?")[0]];
    if (route) {
      Promise.resolve(route()).then(
        value => res.writeHead(200, {
          "Content-Type": "application/json; charset=utf-8",
          "Cache-Control": "no-store"
        }).end(JSON.stringify(value ?? null)),
        () => res.writeHead(500).end("{}")
      );
      return;
    }

    const mount = mounted.find(m => m.prefix === "/" || url.startsWith(m.prefix));
    if (!mount) {
      res.writeHead(404).end();
      return;
    }

    const rest = mount.prefix === "/" ? url : url.slice(mount.prefix.length - 1);
    const file = resolveInside(mount.base, rest);
    if (!file) {
      res.writeHead(403).end();
      return;
    }

    fs.stat(file, (err, stat) => {
      if (err || !stat.isFile()) {
        res.writeHead(404, { "Content-Type": "text/plain" }).end("not found");
        return;
      }

      const headers = {
        "Content-Type": TYPES[path.extname(file).toLowerCase()] ||
                        "application/octet-stream",
        "Content-Length": stat.size,
        // The files live on this disk and are read fresh every launch.
        // Caching them would only ever hide an update.
        "Cache-Control": "no-store"
      };

      if (mount.cors) headers["Access-Control-Allow-Origin"] = "*";

      if (req.method === "HEAD") {
        res.writeHead(200, headers).end();
        return;
      }

      res.writeHead(200, headers);
      fs.createReadStream(file).pipe(res);
    });
  });

  return new Promise((resolve, reject) => {
    server.once("error", reject);
    // Port 0 means "whichever one is free".
    server.listen(0, "127.0.0.1", () => {
      const { port } = server.address();
      resolve({
        url: `http://127.0.0.1:${port}`,
        port,
        close: () => new Promise(done => server.close(done))
      });
    });
  });
}

module.exports = { serve, resolveInside, TYPES };

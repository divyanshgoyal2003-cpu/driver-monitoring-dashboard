/*
 * Where the desktop build gets a location.
 *
 * Chromium resolves navigator.geolocation through a Google service
 * that wants an API key. A packaged Electron app has no key, so the
 * call fails with POSITION_UNAVAILABLE and the forecast -- one of the
 * inputs the risk weighting uses -- never arrives.
 *
 * Windows already knows where it is, and will say so through the
 * location service the user controls in Settings > Privacy. That costs
 * no key, no account, and no third party: nothing leaves the machine
 * to answer this. If the service is off or the user has denied
 * location to desktop apps, this returns null and the app falls back
 * to the browser's own geolocation, which fails exactly as it would
 * have anyway.
 *
 * What the coordinates are then used for is unchanged: docs/weather.js
 * rounds them to two decimals and asks Open-Meteo for a forecast, only
 * after the driver has clicked Locate.
 */

const { spawn } = require("node:child_process");

// GeoCoordinateWatcher resolves asynchronously -- the first reading is
// usually Unknown while the service warms up, so this waits for Ready
// rather than reading once and giving up.
const SCRIPT = `
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Device
$w = New-Object System.Device.Location.GeoCoordinateWatcher
$w.Start()
$n = 0
while (($w.Status -ne 'Ready') -and ($n -lt 40)) { Start-Sleep -Milliseconds 250; $n++ }
$p = $w.Position
if ($p -and -not $p.Location.IsUnknown) {
  "$($p.Location.Latitude),$($p.Location.Longitude),$($p.Location.HorizontalAccuracy)"
}
$w.Stop()
`;

const TIMEOUT = 15000;

// Asked once per launch. A parked car does not move, and a forecast
// resolves to about a kilometre anyway.
let remembered = null;

function where() {
  if (remembered) return Promise.resolve(remembered);

  return new Promise(resolve => {
    let done = false;
    const finish = value => {
      if (done) return;
      done = true;
      if (value) remembered = value;
      resolve(value);
    };

    let child;
    try {
      child = spawn("powershell.exe", [
        "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
        "-Command", SCRIPT
      ], { windowsHide: true });
    } catch (e) {
      finish(null);
      return;
    }

    const timer = setTimeout(() => { try { child.kill(); } catch (e) {} finish(null); },
                             TIMEOUT);

    let out = "";
    child.stdout.on("data", d => { out += d; });
    child.on("error", () => { clearTimeout(timer); finish(null); });
    child.on("close", () => {
      clearTimeout(timer);

      const [lat, lon, accuracy] = out.trim().split(",").map(Number);
      const sane = Number.isFinite(lat) && Number.isFinite(lon) &&
                   Math.abs(lat) <= 90 && Math.abs(lon) <= 180 &&
                   !(lat === 0 && lon === 0);

      finish(sane
        ? { latitude: lat, longitude: lon,
            accuracy: Number.isFinite(accuracy) ? accuracy : 1000 }
        : null);
    });
  });
}

module.exports = { where };

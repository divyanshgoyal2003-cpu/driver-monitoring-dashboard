<!--
  The body of the GitHub release for the desktop build. Paste it whole
  into the release's notes box.

  Two things that cost an afternoon the first time:

  - Copy the RAW file, never a rendered preview of it. Copying rendered
    markdown hands you the words without the syntax, and the result
    looks fine in the box while every heading and bold run is gone.
  - Do not pipe it through clip.exe. That converts to the Windows ANSI
    codepage, which has no em dash, so all twelve become "?". Use:

        Get-Content desktop\RELEASE_NOTES.md -Raw -Encoding UTF8 | Set-Clipboard

  The SHA256 below belongs to one specific build. Any rebuild changes
  it, so regenerate it with Get-FileHash and update the line, or do not
  publish a hash at all -- a wrong one is worse than none.
-->

Driver fatigue and distraction monitoring that runs entirely on the
machine you open it on. No install, no Python, no account, and no
network — the camera feed is processed locally and there is no server
to send it to.

**Download `DriverMonitor-1.0.0-portable.exe` and double-click it.**

Windows will show a blue "Windows protected your PC" box, because the
executable is not code-signed — a certificate costs a few hundred
dollars a year and this is a student project. Click **More info**, then
**Run anyway**. If you would rather not, the same app runs in any
browser at
<https://divyanshgoyal2003-cpu.github.io/driver-monitoring-dashboard/>.

### What it does

- **Microsleep** — eyes closed past 1.5 s
- **Drowsiness** — PERCLOS over a rolling minute
- **Yawning** — counted when the mouth closes again
- **Distraction** — head off the road past 2 s
- **Seatbelt** — the strap's two edges, found by hand rather than by a model
- **Expression** — brow and mouth shape, smoothed
- **Camera** — a covered lens, and a feed that has stopped moving
- **Hands on the wheel** and **phone in your hand** — optional, one model each
- **Safety index** — each lapse weighed by how bad it is, how trustworthy
  the measurement is, and what the hour and the weather are doing to the
  cost of it

Named drivers each keep their own learned eye threshold and drive log,
stored on the device and nowhere else.

### It is the web app, not a port of it

The window contains the same `docs/` folder the website serves, handed
to it over a loopback server inside the process. One copy of the
detectors, the scoring and the interface — there is no desktop version
that can drift from the one in the browser.

The MediaPipe runtime and the webfont are bundled rather than fetched,
and the location comes from the Windows location service instead of the
Google endpoint Chromium cannot reach without an API key. Those are the
only two things the shell supplies; everything else is the web app.

### Checking it yourself

    DriverMonitor-1.0.0-portable.exe --self-test

Runs the project's 131 checks plus the model, camera, location, font,
offline and unsupported-dialog checks inside the shipped executable, and
shows the verdict in a dialog. Nothing is printed to a terminal: a packaged Electron app is a
GUI binary and attaches to no console, which is why the result is shown
rather than written out.

To see the 131 detector checks themselves, press **Alt** in the running
app and pick *App > Self-test*.

### Verifying the download

    SHA256: 73120b29df467044d5e5f5d8e7cbddc93f528d9b94376e86465b5dd92b06f0dd

In PowerShell:

    Get-FileHash .\DriverMonitor-1.0.0-portable.exe -Algorithm SHA256

### Requirements

Windows 10 or 11, 64-bit, and a camera. Roughly 87 MB to download and
about 190 MB unpacked to a temporary folder while it runs.

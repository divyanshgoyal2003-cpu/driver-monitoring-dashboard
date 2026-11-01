"""
Record a scripted drive, and write down what was asked for.

    python -m evaluation.record_session --video drive.mp4 --labels drive.csv

The evaluation harness can only measure the monitor against ground
truth, and writing ground truth by hand means scrubbing a video with a
notepad, guessing at the frame a yawn ended. That is slow enough that
it does not get done, which is why the detectors have never been
measured.

This removes the excuse. It prompts the driver through a fixed script
-- look away, hold a phone up, close your eyes, drop your hands -- and
writes the label file from the same schedule it drove. Recording and
labelling become one pass.

**The labels say what the driver was asked to do, not what they did.**
Nobody times a yawn to the frame, cues get missed, and a hand wanders
back into shot early. So the CSV this writes is a first draft to be
corrected against the video, not ground truth on its own. It is
honest about that in a comment at the top of the file, because a
measurement trusted further than it deserves is worse than none.

Two things it deliberately does not do:

  - **The cue is never burned into the recording.** It is drawn on the
    preview window only. Text painted onto the frame would change the
    brightness and the variance the obstruction and low-light
    detectors read, so the recording would no longer be the thing
    being measured.

  - **It does not label blinks or yawns.** Those are point events a
    tenth of a second long; a cue card cannot place them. The CSV
    names the labels it does cover, and the harness scores only those.
"""

import argparse
import os
import subprocess
import sys
import threading
import time

import cv2

from app.config import CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT


class Step:
    """One cue: how long to hold it, what to say, what it means."""

    def __init__(self, seconds, cue, label=None, spoken=None, note=""):

        self.seconds = seconds
        self.cue = cue
        self.label = label
        self.note = note

        # The written cue is cut to fit a banner. Spoken aloud it can
        # afford to be a full sentence, and some of them need to be:
        # "drop your hands" is ambiguous until you add "out of shot".
        self._spoken = spoken

    @property
    def spoken(self):
        return self._spoken or self.cue


# The cues are long on purpose. Every episode detector waits before it
# commits -- two seconds of gaze away before DISTRACTED, three with no
# hands before HANDS OFF WHEEL -- so a four-second cue would be mostly
# dwell time and would score as a miss for reasons that have nothing to
# do with accuracy.
SCRIPT = (
    Step(10.0, "Sit still. Look straight at the camera.",
         spoken="Recording. Sit still and look straight at the camera.",
         note="calibration - the head pose baseline is learned here"),
    Step(10.0, "Keep looking ahead. Blink normally.",
         spoken="Keep looking ahead, and blink normally.",
         note="a quiet stretch, so false positives have room to show"),

    Step(10.0, "Turn your head right, away from the camera.", "distracted",
         spoken="Now turn your head to the right, away from the camera, "
                "and hold it there."),
    Step(6.0, "Look straight ahead again.",
         spoken="Look straight ahead again."),

    Step(10.0, "Hold a phone up beside your face.", "phone",
         spoken="Hold a phone up beside your face, and keep it in shot."),
    Step(6.0, "Put the phone down. Look ahead.",
         spoken="Put the phone down, and look ahead."),

    Step(12.0, "Yawn twice, slowly and wide.",
         spoken="Yawn twice, slowly and wide. Close your mouth fully in "
                "between them.",
         note="not labelled - a yawn is too short for a cue to place"),

    Step(10.0, "Close your eyes. Keep them closed.", "microsleep",
         spoken="Close your eyes now, and keep them closed until I tell "
                "you to open them."),
    Step(6.0, "Open your eyes. Look ahead.",
         spoken="Open your eyes, and look ahead."),

    Step(12.0, "Hold both hands up where the camera can see them.",
         spoken="Hold both hands up, palms towards the camera, where it "
                "can see them.",
         note="the wheel monitor needs this before absence means "
              "anything"),
    Step(10.0, "Drop both hands out of shot.", "hands_off",
         spoken="Drop both hands right down, out of the picture."),
    Step(6.0, "Hands back up where they can be seen.",
         spoken="Hands back up where the camera can see them."),

    Step(10.0, "Cover the lens with your palm.", "camera_blocked",
         spoken="Cover the lens with your palm. Use your hand, not the "
                "camera switch or the shutter."),
    Step(6.0, "Take your hand off the lens. Look ahead.",
         spoken="Take your hand off the lens, and look ahead."),
)

# Spoken this long before a cue ends, so somebody with their eyes shut
# knows the change is coming rather than being startled by it.
COUNTDOWN_AT = 3.5

# How many frames to sample before opening the writer. A webcam's rate
# settles after it has woken up, and the file's frame rate has to match
# the real one or every timestamp in the recording is wrong.
WARMUP_FRAMES = 40

# Beyond this much disagreement between the rate written and the rate
# achieved, the timestamps are far enough out to say so.
FPS_TOLERANCE = 0.02

BANNER = (0, 0, 0)
TEXT = (255, 255, 255)
COUNTDOWN = (120, 220, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX


# One long-lived PowerShell reading cues off its stdin. Speaking
# through a fresh process per cue costs half a second of startup, which
# is enough to drift the cue past the moment it is describing.
SPEECH = (
    "Add-Type -AssemblyName System.Speech; "
    "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
    "$s.Rate = 1; "
    "while (($line = [Console]::In.ReadLine()) -ne $null) { "
    "if ($line.Length -gt 0) { $s.Speak($line) } }"
)


class Voice:
    """
    Spoken cues, because a cue you cannot look at is not a cue.

    Half the script asks the driver to stop watching the screen -- eyes
    closed, head turned away, a palm over the lens -- and during those
    the countdown in the corner may as well not be there. Windows can
    speak without any package being installed, so it does.

    Silence is always an acceptable outcome. If speech cannot start, or
    dies partway, the recording carries on without it rather than
    taking the session down.
    """

    def __init__(self, enabled=True):

        self.process = None
        self.reason = None

        if not enabled:
            self.reason = "turned off"
            return

        try:
            self.process = subprocess.Popen(
                ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
                 "-Command", SPEECH],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=True,
            )
        except OSError as error:
            self.reason = f"could not start PowerShell ({error})"

    @property
    def available(self):
        return self.process is not None and self.process.poll() is None

    def say(self, text):

        if not self.available:
            return

        try:
            self.process.stdin.write(text + "\n")
            self.process.stdin.flush()
        except (OSError, ValueError):
            # It went away mid-session. Carry on quietly.
            self.process = None
            self.reason = "speech stopped partway"

    def close(self):

        if self.process is None:
            return

        try:
            self.process.stdin.close()
        except (OSError, ValueError):
            pass

        try:
            self.process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.process.kill()

        self.process = None


def chime(frequency=880, milliseconds=120):
    """
    A marker at each change of cue, on a thread.

    winsound.Beep blocks for its whole duration, and blocking the
    capture loop drops frames -- which would bend the timestamps the
    labels are built from.
    """

    def play():
        try:
            import winsound

            winsound.Beep(frequency, milliseconds)
        except Exception:
            pass

    threading.Thread(target=play, daemon=True).start()


def covered_labels(script=SCRIPT):
    """The interval labels this script actually exercises."""

    seen = []

    for step in script:
        if step.label and step.label not in seen:
            seen.append(step.label)

    return seen


def total_seconds(script=SCRIPT):
    return sum(step.seconds for step in script)


def build_labels(boundaries, fps, script=SCRIPT):
    """
    Turn recorded phase boundaries into ground truth.

    `boundaries` is the frame index each step began at, with the total
    frame count appended. Times come from frame indices rather than
    from the wall clock, so they land in the same timebase the harness
    reads the video in -- CameraManager derives a file's timestamps as
    frame / fps, and these are built the same way.
    """

    rows = []

    for index, step in enumerate(script):

        if step.label is None or index + 1 >= len(boundaries):
            continue

        start = boundaries[index] / fps
        end = boundaries[index + 1] / fps

        # A step the recording was cut short in leaves nothing to
        # measure.
        if end - start < 1e-6:
            continue

        rows.append((start, end, step.label))

    return rows


def format_labels(rows, covered, complete=True):
    """The CSV text, with the caveats it needs at the top."""

    lines = [
        "# Written by evaluation/record_session.py from the cue",
        "# schedule: these are the times the driver was ASKED to do",
        "# each thing, not the times they did it. Watch the video and",
        "# correct them before trusting the numbers.",
        "#",
        "# Blinks and yawns are not here. They are too short for a cue",
        "# to place, so they have to be added by hand. The cover line",
        "# below keeps the harness from scoring what nobody labelled.",
    ]

    if not complete:
        lines += [
            "#",
            "# NOTE: the recording was stopped early, so the later",
            "# cues in the script never happened.",
        ]

    lines += [
        "",
        f"#cover: {', '.join(covered)}",
        "",
        "start,end,label",
    ]

    lines += [f"{start:.2f},{end:.2f},{label}" for start, end, label in rows]

    return "\n".join(lines) + "\n"


def measure_fps(capture, frames=WARMUP_FRAMES):
    """
    The rate the camera actually delivers, not the one it advertises.

    Returns (fps, first_frame). The frame is handed back so the warmup
    does not throw away footage the driver is already sitting for.
    """

    first = None
    captured = 0
    started = None

    while captured < frames:
        success, frame = capture.read()

        if not success:
            break

        if first is None:
            first = frame
            started = time.perf_counter()

        captured += 1

    elapsed = time.perf_counter() - started if started else 0.0

    if captured < 2 or elapsed <= 0:
        return None, first

    return (captured - 1) / elapsed, first


def draw_cue(frame, step, remaining, elapsed, total):
    """
    The preview overlay. Drawn on a copy that is never recorded.
    """

    preview = frame.copy()
    height, width = preview.shape[:2]

    banner = preview.copy()
    cv2.rectangle(banner, (0, height - 96), (width, height), BANNER, -1)
    cv2.addWeighted(banner, 0.65, preview, 0.35, 0, preview)

    cv2.putText(preview, step.cue, (16, height - 58), FONT, 0.62, TEXT, 2,
                cv2.LINE_AA)

    cv2.putText(preview, f"{remaining:.0f}s", (16, height - 22), FONT,
                0.7, COUNTDOWN, 2, cv2.LINE_AA)

    cv2.putText(preview, f"{elapsed:.0f} / {total:.0f}s   Q to stop",
                (90, height - 22), FONT, 0.5, TEXT, 1, cv2.LINE_AA)

    # A progress bar along the very top, so the driver can see the end
    # coming without reading anything.
    done = int(width * min(elapsed / total, 1.0)) if total else 0
    cv2.rectangle(preview, (0, 0), (done, 5), COUNTDOWN, -1)

    return preview


def record(video_path, source=CAMERA_INDEX, width=FRAME_WIDTH,
           height=FRAME_HEIGHT, script=SCRIPT, preview=True, voice=None):
    """
    Run the script, writing clean frames to `video_path`.

    Returns (boundaries, fps, frames, complete).
    """

    capture = cv2.VideoCapture(source)

    if not capture.isOpened():
        raise RuntimeError(f"Unable to open camera: {source}")

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    writer = None
    boundaries = []
    frames = 0
    complete = True

    try:
        fps, frame = measure_fps(capture)

        if frame is None:
            raise RuntimeError("The camera opened but delivered no frames")

        if fps is None:
            fps = 20.0

        size = (frame.shape[1], frame.shape[0])
        writer = cv2.VideoWriter(
            video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size
        )

        if not writer.isOpened():
            raise RuntimeError(f"Unable to write video to {video_path}")

        total = total_seconds(script)
        started = time.perf_counter()
        step_started = started

        for step in script:
            boundaries.append(frames)
            step_started = time.perf_counter()

            chime()

            if voice is not None:
                voice.say(step.spoken)

            counted = False

            while True:
                now = time.perf_counter()
                remaining = step.seconds - (now - step_started)

                if remaining <= 0:
                    break

                # Long enough for a warning to be worth giving, and
                # for it to finish before the cue changes.
                if (voice is not None and not counted
                        and step.seconds > COUNTDOWN_AT + 3.0
                        and remaining <= COUNTDOWN_AT):
                    voice.say("three, two, one")
                    counted = True

                success, frame = capture.read()

                if not success:
                    complete = False
                    break

                # The recording gets the frame as the camera saw it.
                writer.write(frame)
                frames += 1

                if preview:
                    cv2.imshow(
                        "Recording - evaluation session",
                        draw_cue(frame, step, remaining,
                                 now - started, total)
                    )

                    if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
                        complete = False
                        break

            if not complete:
                break

        boundaries.append(frames)

        achieved = frames / (time.perf_counter() - started)

    finally:
        capture.release()

        if writer is not None:
            writer.release()

        if preview:
            cv2.destroyAllWindows()

    return boundaries, fps, frames, complete, achieved


def parse_args(argv=None):

    parser = argparse.ArgumentParser(
        description="Record a scripted drive and write its ground truth."
    )

    parser.add_argument("--video", default="drive.mp4",
                        help="where to write the recording")
    parser.add_argument("--labels", default="drive.csv",
                        help="where to write the ground truth")
    parser.add_argument("--source", default=CAMERA_INDEX,
                        help="camera index (default 0)")
    parser.add_argument("--no-preview", action="store_true",
                        help="do not open a window. With --no-voice too "
                             "the cues are invisible, which is only "
                             "useful for testing the plumbing")
    parser.add_argument("--no-voice", action="store_true",
                        help="do not speak the cues. Half the script "
                             "asks you to stop watching the screen, so "
                             "this is worth keeping on")

    args = parser.parse_args(argv)

    try:
        args.source = int(args.source)
    except (TypeError, ValueError):
        pass

    return args


def main(argv=None):

    args = parse_args(argv)

    # Failing on a missing folder after the driver has sat down and
    # started performing would be a poor way to find out.
    for path in (args.video, args.labels):
        folder = os.path.dirname(path)

        if folder:
            os.makedirs(folder, exist_ok=True)

    voice = Voice(enabled=not args.no_voice)

    print(f"Recording {total_seconds():.0f}s to {args.video}")
    print("Follow the cue at the bottom of the window. Q stops early.")

    if voice.available:
        print("Cues are spoken, so you can close your eyes and look away.")
    elif not args.no_voice:
        print(f"No spoken cues ({voice.reason}); watch the window.")

    print()

    try:
        boundaries, fps, frames, complete, achieved = record(
            args.video, source=args.source, preview=not args.no_preview,
            voice=voice
        )
    except RuntimeError as error:
        print(error, file=sys.stderr)
        return 1
    finally:
        voice.close()

    if not frames:
        print("Nothing was recorded.", file=sys.stderr)
        return 1

    rows = build_labels(boundaries, fps)

    with open(args.labels, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(format_labels(rows, covered_labels(), complete))

    print(f"  {args.video}   {frames} frames at {fps:.1f} fps "
          f"({frames / fps:.0f}s)")
    print(f"  {args.labels}   {len(rows)} episodes")

    # The file's frame rate is what every timestamp in the evaluation is
    # derived from. If the camera did not keep up with it, the labels
    # and the video drift apart and the report is measuring the drift.
    if fps and abs(achieved - fps) / fps > FPS_TOLERANCE:
        print()
        print(f"  WARNING: the camera was measured at {fps:.1f} fps but "
              f"delivered {achieved:.1f} fps.")
        print("  Timestamps in the recording are stretched by roughly "
              f"{abs(achieved - fps) / fps * 100:.0f}%. Re-record in "
              "better light,")
        print("  or correct the label times against the video before "
              "trusting them.")

    if not complete:
        print()
        print("  Stopped early: the later cues never happened.")

    print()
    print("Now watch the recording and correct the label times, then:")
    print(f"  python -m evaluation.evaluate --video {args.video} "
          f"--labels {args.labels}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

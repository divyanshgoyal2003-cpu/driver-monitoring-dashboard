"""
Measure the monitor against hand-written ground truth.

    python -m evaluation.evaluate --video drive.mp4 --labels drive.csv

Runs the real pipeline over a recorded video and answers two questions
the project could not answer before:

  1. How accurate is each detector? Precision and recall against what a
     human says actually happened (see evaluation/labels.py for the
     format). Without this the system can report what it found but not
     whether it was right.

  2. How differently do the two scoring engines read the same drive?
     Both are run over the same frames in a single pass, so the
     comparison costs nothing extra and neither engine can be accused
     of having had an easier video.

Labels are optional. Without them only the second half runs, which is
still useful for comparing the engines.
"""

import argparse
import sys

from app.camera import CameraManager
from app.config import SAFE_SCORE, CAUTION_SCORE, WEATHER_MULTIPLIERS
from app.video_processor import VideoProcessor
from evaluation import labels as labels_module
from evaluation.matching import (
    match_points,
    match_intervals,
    spans_from_flags
)
from risk_engine.context import ContextProvider
from risk_engine.scoring import DriverSafetyIndex

# How far a detection may sit from its labelled time and still count.
# Hand-labelling is not millisecond-accurate, and at 18 FPS a frame is
# already 55 ms.
POINT_TOLERANCE = 0.5

# state -> the per-sample boolean for each episode type.
EPISODE_FLAGS = {
    "microsleep": lambda s: bool(
        (s.get("drowsiness") or {}).get("microsleep")
    ),
    "distracted": lambda s: bool(
        (s.get("distraction") or {}).get("distracted")
    ),
    "phone": lambda s: bool((s.get("phone") or {}).get("in_use")),
    "no_seatbelt": lambda s: (
        (s.get("seatbelt") or {}).get("status") == "OFF"
    ),
    "camera_blocked": lambda s: bool((s.get("camera") or {}).get("blocked")),
    "hands_off": lambda s: bool((s.get("wheel") or {}).get("hands_off")),
}

# state -> the running count whose increments mark a point event.
POINT_COUNTS = {
    "blink": ("blink", "blink_count"),
    "yawn": ("yawn", "yawn_count"),
}


class Recording:
    """What one pass over a video saw."""

    def __init__(self):

        self.samples = []
        self.points = {name: [] for name in POINT_COUNTS}
        self.flags = {name: [] for name in EPISODE_FLAGS}
        self.scores = {"dynamic": [], "static": []}

        self._last_count = {name: 0 for name in POINT_COUNTS}

    def add(self, state, timestamp, dynamic_risk, static_risk):

        self.samples.append(timestamp)

        for name, (section, key) in POINT_COUNTS.items():
            reading = state.get(section)
            count = None if reading is None else reading.get(key)

            # No reading at all. The detector returns None rather than
            # a dict when it has no face to measure, and its counter is
            # untouched underneath -- so the total is held, not read as
            # zero. Reading it as zero made the first frame after every
            # face gap look like the whole session's blinks happening
            # at once, which is where a 51/min blink rate came from.
            if count is None:
                continue

            # A counter that jumps by more than one means several
            # events landed between samples; record them all.
            for _ in range(count - self._last_count[name]):
                self.points[name].append(timestamp)

            self._last_count[name] = count

        for name, read in EPISODE_FLAGS.items():
            self.flags[name].append(read(state))

        self.scores["dynamic"].append(dynamic_risk["score"])
        self.scores["static"].append(static_risk["score"])

    def spans(self, name):
        return spans_from_flags(self.samples, self.flags[name])

    @property
    def duration(self):
        return self.samples[-1] - self.samples[0] if self.samples else 0.0


def analyse(video, hour=None, weather=None, speed=None, progress=None,
            enhance=True):
    """
    One pass over `video`, scoring every frame with both engines.
    """

    camera = CameraManager(video)
    provider = ContextProvider(weather=weather, speed=speed, hour=hour)

    # The processor owns the dynamic engine; a second, static one is
    # fed the same states so both see identical frames.
    processor = VideoProcessor(dynamic_weights=True,
                               context_provider=provider,
                               enhance=enhance)
    static = DriverSafetyIndex(dynamic=False)

    recording = Recording()
    frames = 0

    try:
        while True:
            frame = camera.read_frame()

            if frame is None:
                break

            timestamp = camera.last_timestamp

            processor.process(frame, timestamp, render=False)
            state = processor.state

            static_risk = static.update(
                state, timestamp, state["risk"]["context"]
            )

            recording.add(state, timestamp, state["risk"], static_risk)

            frames += 1

            if progress and frames % 50 == 0:
                progress(frames, timestamp)

    finally:
        camera.release()
        processor.close()

    return recording


def score_against(recording, truth):
    """
    Precision and recall per label. Returns [(name, Score)].

    Labels the file did not claim to cover are left out rather than
    scored against nothing: an unlabelled detector would otherwise
    report 0% precision and read as a broken one.
    """

    results = []
    covers = getattr(truth, "covers", None)

    def scored(name):
        return covers is None or name in covers

    for name in labels_module.POINT_LABELS:
        if not scored(name):
            continue

        expected = labels_module.of_type(truth, name)

        if not expected and not recording.points[name]:
            continue

        score = match_points(
            recording.points[name], expected, POINT_TOLERANCE
        )
        score.label = name
        results.append((name, score))

    for name in labels_module.INTERVAL_LABELS:
        if not scored(name):
            continue

        expected = labels_module.of_type(truth, name)

        if not expected and not any(recording.flags[name]):
            continue

        score = match_intervals(
            recording.samples, recording.spans(name), expected
        )
        score.label = name
        results.append((name, score))

    return results


# ----------------------------------------------------------------------
# reporting
# ----------------------------------------------------------------------


def _percent(value):
    return "    --" if value is None else f"{value * 100:5.1f}%"


def format_accuracy(results, recording=None, covers=None, truth=None):

    lines = [
        "Detection accuracy",
        "",
        f"  {'label':<14}{'found':>8}{'actual':>8}"
        f"{'precision':>11}{'recall':>9}{'F1':>8}   unit",
    ]

    for name, score in results:
        lines.append(
            f"  {name:<14}{score.detected:>8}{score.actual:>8}"
            f"{_percent(score.precision):>11}{_percent(score.recall):>9}"
            f"{_percent(score.f1):>8}   {score.unit}"
        )

    if not results:
        lines.append("  (nothing labelled and nothing detected)")

    skipped = _unmeasured(recording, covers, truth)

    if skipped:
        # Naming these matters. A reader who does not find blink in the
        # table above could reasonably conclude the detector found
        # nothing, when the truth is that nobody checked it.
        lines += [
            "",
            "  Not measured, because the labels do not cover them:",
            f"    {', '.join(skipped)}",
        ]

    return "\n".join(lines)


def _unmeasured(recording, covers, truth=None):
    """
    Labels left out of the scoring, named so the gap cannot pass for a
    clean sheet. A label nobody looked for and nothing found is not
    worth a line -- but one the file *writes down* and then excludes
    is, however little was detected, because a cue missing from both
    the table and this list reads as a cue that never happened.

    That case is not hypothetical. take2.csv records an eleven second
    hands_off episode and leaves the label out of its cover line,
    because the clip was filmed at a desk and there is no wheel in it
    to have hands off; the monitor correctly reported nothing, and
    without this the whole cue vanished from the report.
    """

    if recording is None or covers is None:
        return []

    performed = {
        label.label for label in (truth or [])
    }

    names = []

    for name in labels_module.POINT_LABELS:
        if name in covers:
            continue

        found = len(recording.points[name])

        if found or name in performed:
            names.append(f"{name} ({found} detected)")

    for name in labels_module.INTERVAL_LABELS:
        if name in covers:
            continue

        episodes = len(recording.spans(name))

        if episodes or name in performed:
            names.append(f"{name} ({episodes} detected)")

    return names


def format_detections(recording):
    """What was found, when there is no ground truth to judge it by."""

    lines = ["What the monitor detected", ""]

    for name in labels_module.POINT_LABELS:
        found = len(recording.points[name])
        rate = (
            f"   ({found / recording.duration * 60:.1f}/min)"
            if recording.duration > 0 else ""
        )
        lines.append(f"  {name:<14}{found:>6}{rate}")

    for name in labels_module.INTERVAL_LABELS:
        spans = recording.spans(name)
        seconds = sum(end - start for start, end in spans)
        lines.append(
            f"  {name:<14}{len(spans):>6} episodes, {seconds:6.1f} s"
        )

    return "\n".join(lines)


def format_comparison(recording):

    lines = [
        "Scoring engines, same frames",
        "",
        f"  {'engine':<10}{'mean':>8}{'lowest':>9}"
        f"{'% CAUTION':>12}{'% DANGER':>11}",
    ]

    for engine in ("dynamic", "static"):
        scores = recording.scores[engine]

        if not scores:
            continue

        caution = sum(
            1 for s in scores if CAUTION_SCORE <= s < SAFE_SCORE
        ) / len(scores)
        danger = sum(1 for s in scores if s < CAUTION_SCORE) / len(scores)

        lines.append(
            f"  {engine:<10}{sum(scores) / len(scores):>8.1f}"
            f"{min(scores):>9.1f}{_percent(caution):>12}"
            f"{_percent(danger):>11}"
        )

    return "\n".join(lines)


def parse_args(argv=None):

    parser = argparse.ArgumentParser(
        description="Measure the monitor against hand-written ground "
                    "truth, and compare the two scoring engines."
    )

    parser.add_argument("--video", required=True,
                        help="recorded drive to analyse")
    parser.add_argument("--labels",
                        help="ground truth CSV (see evaluation/labels.py)")
    parser.add_argument("--hour", type=int, choices=range(24),
                        metavar="0-23",
                        help="score as if the drive happened at this hour")
    parser.add_argument("--weather",
                        choices=sorted(WEATHER_MULTIPLIERS),
                        help="score as if the drive happened in these "
                             "conditions. No forecast is fetched here: "
                             "a recording's weather was whatever it was "
                             "at the time, not whatever it is now")
    parser.add_argument("--speed", type=float)
    parser.add_argument("--no-enhance", action="store_true",
                        help="do not lift dim frames before detection. "
                             "Run the same video twice, with and "
                             "without, to measure what it is worth")
    parser.add_argument("--quiet", action="store_true",
                        help="no progress while the video is read")

    return parser.parse_args(argv)


def main(argv=None):

    args = parse_args(argv)

    truth = None

    # Read the labels first: a typo in the CSV should not cost several
    # minutes of video analysis before it is reported.
    if args.labels:
        try:
            truth = labels_module.parse_file(args.labels)
        except (OSError, labels_module.LabelError) as error:
            print(f"Could not read {args.labels}: {error}",
                  file=sys.stderr)
            return 1

    # Padded to a fixed width so the next line fully overwrites it,
    # however long the counts get.
    def progress(frames, timestamp):
        print(f"\r  {frames} frames, {timestamp:.0f}s of video...".ljust(60),
              end="", file=sys.stderr)

    try:
        recording = analyse(
            args.video,
            hour=args.hour,
            weather=args.weather,
            speed=args.speed,
            progress=None if args.quiet else progress,
            enhance=not args.no_enhance
        )
    except RuntimeError as error:
        # A mistyped filename is the usual cause, and deserves a line
        # rather than a traceback.
        print(error, file=sys.stderr)
        return 1

    if not args.quiet:
        print("\r" + " " * 60 + "\r", end="", file=sys.stderr)

    if not recording.samples:
        print(f"No frames read from {args.video}", file=sys.stderr)
        return 1

    print(f"{args.video}  -  {len(recording.samples)} frames, "
          f"{recording.duration:.1f} s")
    print()

    if truth is None:
        print(format_detections(recording))
        print()
        print("  No --labels given, so there is nothing to check these")
        print("  against. See evaluation/labels.py for the format.")
    else:
        print(format_accuracy(
            score_against(recording, truth),
            recording,
            getattr(truth, "covers", None),
            truth
        ))

    print()
    print(format_comparison(recording))

    return 0


if __name__ == "__main__":
    sys.exit(main())

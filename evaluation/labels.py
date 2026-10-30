"""
Hand-written ground truth for a recorded drive.

The project can say what it detected but not whether it was right. To
measure that, someone has to watch the video and write down what really
happened. This is the format they write it in, kept as plain CSV so it
can be typed in a text editor or a spreadsheet while scrubbing through
the footage:

    start,end,label
    12.3,,blink
    15.1,,blink
    20.0,26.5,distracted
    40.2,48.0,phone
    # anything after a hash is ignored

Times are seconds from the start of the video, matching the timestamps
CameraManager derives for a file. Point events (a blink, a yawn) need
only a start. Episodes (looking away, holding a phone) need both ends.
"""

import csv


# Things that happen at an instant, matched against detections within a
# tolerance window.
POINT_LABELS = ("blink", "yawn")

# Things that last, matched by how much of the time the two agree.
INTERVAL_LABELS = ("microsleep", "distracted", "phone", "no_seatbelt")

LABELS = POINT_LABELS + INTERVAL_LABELS


class LabelError(ValueError):
    """A ground truth file that cannot be read as written."""


class Label:

    def __init__(self, label, start, end=None):

        self.label = label
        self.start = start
        self.end = end

    @property
    def is_point(self):
        return self.end is None

    def covers(self, timestamp):
        """Whether this episode is running at `timestamp`."""

        if self.is_point:
            return False

        return self.start <= timestamp < self.end

    def __repr__(self):

        span = (
            f"{self.start:.2f}"
            if self.is_point
            else f"{self.start:.2f}-{self.end:.2f}"
        )

        return f"<{self.label} {span}>"


def parse(lines):
    """
    Read ground truth from any iterable of CSV lines. Returns a list of
    Label, sorted by start time.
    """

    labels = []

    rows = csv.reader(
        line for line in lines
        if line.strip() and not line.lstrip().startswith("#")
    )

    for number, row in enumerate(rows, start=1):

        if not row:
            continue

        cells = [cell.strip() for cell in row]

        # Tolerate the header row, whether or not it is there.
        if number == 1 and cells[0].lower() == "start":
            continue

        if len(cells) < 3:
            raise LabelError(
                f"line {number}: expected start,end,label - got {row!r}"
            )

        start, end, name = cells[0], cells[1], cells[2].lower()

        if name not in LABELS:
            raise LabelError(
                f"line {number}: unknown label {name!r}. "
                f"Expected one of {', '.join(LABELS)}"
            )

        labels.append(_build(number, name, start, end))

    return sorted(labels, key=lambda label: label.start)


def parse_file(path):

    with open(path, newline="", encoding="utf-8") as handle:
        return parse(handle)


def _build(number, name, start, end):

    start_time = _time(number, start, "start")

    if name in POINT_LABELS:
        if end:
            raise LabelError(
                f"line {number}: {name} happens at an instant, so it "
                f"takes no end time"
            )

        return Label(name, start_time)

    if not end:
        raise LabelError(
            f"line {number}: {name} lasts, so it needs an end time"
        )

    end_time = _time(number, end, "end")

    if end_time <= start_time:
        raise LabelError(
            f"line {number}: end {end_time} is not after start "
            f"{start_time}"
        )

    return Label(name, start_time, end_time)


def _time(number, text, which):

    try:
        value = float(text)
    except ValueError:
        raise LabelError(
            f"line {number}: {which} time {text!r} is not a number"
        ) from None

    if value < 0:
        raise LabelError(f"line {number}: {which} time cannot be negative")

    return value


def of_type(labels, name):

    return [label for label in labels if label.label == name]

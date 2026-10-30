"""
Scoring detections against hand-written ground truth.

Two kinds of answer are needed, because the two kinds of label mean
different things:

Point events (a blink) are counted. The question is how many real
blinks were found and how many were invented, so detections are paired
one-to-one with labels inside a tolerance window. The window exists
because nobody hand-labels a video to the millisecond, and because a
blink detected one frame late is not a miss.

Episodes (looking away) are not counted but measured. Two detectors can
agree that a driver looked away while disagreeing about when it started
and stopped, so these are scored on how much of the time the two agree,
sample by sample. "Of the time we called it distraction, how much
really was" is the useful question, and it is what precision means here.
"""


class Score:
    """Precision, recall and F1 for one label."""

    def __init__(self, label, true_positive, false_positive,
                 false_negative, unit="events"):

        self.label = label
        self.true_positive = true_positive
        self.false_positive = false_positive
        self.false_negative = false_negative
        self.unit = unit

    @property
    def detected(self):
        return self.true_positive + self.false_positive

    @property
    def actual(self):
        return self.true_positive + self.false_negative

    @property
    def precision(self):
        """Of what was detected, how much was real."""

        if not self.detected:
            return None

        return self.true_positive / self.detected

    @property
    def recall(self):
        """Of what was real, how much was detected."""

        if not self.actual:
            return None

        return self.true_positive / self.actual

    @property
    def f1(self):

        precision, recall = self.precision, self.recall

        if not precision or not recall:
            return None

        return 2 * precision * recall / (precision + recall)


def match_points(detections, labels, tolerance):
    """
    Pair detected timestamps with labelled ones, nearest first, each
    used at most once.

    Nearest-first rather than in time order: with two labels close
    together, taking them in order can pair the first label with a
    detection that belonged to the second and lose both.
    """

    pairs = sorted(
        (
            (abs(d - label.start), i, j)
            for i, d in enumerate(detections)
            for j, label in enumerate(labels)
            if abs(d - label.start) <= tolerance
        )
    )

    used_detections = set()
    used_labels = set()

    for _, i, j in pairs:
        if i not in used_detections and j not in used_labels:
            used_detections.add(i)
            used_labels.add(j)

    matched = len(used_labels)

    return Score(
        label=None,
        true_positive=matched,
        false_positive=len(detections) - matched,
        false_negative=len(labels) - matched,
    )


def match_intervals(samples, detected_spans, labels):
    """
    Score episodes by agreement at each sample.

    samples:        the timestamps the video was actually analysed at
    detected_spans: (start, end) pairs the system reported
    labels:         Label objects for this episode type
    """

    true_positive = false_positive = false_negative = 0

    for timestamp in samples:

        detected = any(
            start <= timestamp < end for start, end in detected_spans
        )
        actual = any(label.covers(timestamp) for label in labels)

        if detected and actual:
            true_positive += 1
        elif detected:
            false_positive += 1
        elif actual:
            false_negative += 1

    return Score(
        label=None,
        true_positive=true_positive,
        false_positive=false_positive,
        false_negative=false_negative,
        unit="samples",
    )


def spans_from_flags(samples, flags):
    """
    Turn a per-sample boolean into (start, end) spans. The last sample
    closes an open span, so a run that reaches the end of the video is
    not dropped.
    """

    spans = []
    start = None

    for timestamp, flag in zip(samples, flags):

        if flag and start is None:
            start = timestamp

        elif not flag and start is not None:
            spans.append((start, timestamp))
            start = None

    if start is not None and samples:
        spans.append((start, samples[-1]))

    return spans

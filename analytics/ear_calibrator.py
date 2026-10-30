"""
Per-driver eye aspect ratio calibration.

A fixed EAR threshold encodes an assumption about eyelid shape. Eye
aperture varies widely between people: a driver whose open-eye EAR sits
below the fixed threshold is read as permanently closed (PERCLOS pegs at
1.0 and the microsleep alert never stops), while a driver well above it
may never cross it even on a real blink.

This learns the driver's own open and closed EAR over the first few
seconds and places the threshold between them, so the question becomes
"have this driver's eyes narrowed from their normal" rather than "is this
eye narrower than a constant".

The baseline is frozen once learned. A continuously adapting baseline
would drift down with a driver whose eyes slowly narrow over a long
session, normalizing away the drowsiness it exists to detect. Press C to
recalibrate.
"""

import numpy as np

from app.config import (
    EAR_THRESHOLD,
    EAR_CALIBRATION_TIME,
    EAR_OPEN_PERCENTILE,
    EAR_CLOSED_PERCENTILE,
    EAR_CLOSED_FRACTION,
    EAR_OPEN_FRACTION,
    EAR_MIN_RANGE,
    EAR_MIN_SAMPLES,
    EAR_THRESHOLD_MIN,
    EAR_THRESHOLD_MAX
)


class EarCalibrator:
    """
    Collects EAR samples and derives this driver's blink threshold.

    Until calibration finishes, `threshold` is the configured default.
    `confidence` reports how much the eye-based analytics should be
    trusted afterwards; the risk engine scales drowsiness penalties by it,
    so a driver the calibrator could not read well produces softer
    penalties rather than false alarms.
    """

    # confidence by how the threshold was derived
    CONFIDENCE_TWO_ENDED = 1.0
    CONFIDENCE_OPEN_ONLY = 0.8
    CONFIDENCE_DEFAULT = 0.5

    def __init__(self, duration=EAR_CALIBRATION_TIME,
                 default_threshold=EAR_THRESHOLD,
                 open_percentile=EAR_OPEN_PERCENTILE,
                 closed_percentile=EAR_CLOSED_PERCENTILE,
                 closed_fraction=EAR_CLOSED_FRACTION,
                 open_fraction=EAR_OPEN_FRACTION,
                 min_range=EAR_MIN_RANGE,
                 min_samples=EAR_MIN_SAMPLES,
                 threshold_min=EAR_THRESHOLD_MIN,
                 threshold_max=EAR_THRESHOLD_MAX):

        self.duration = duration
        self.default_threshold = default_threshold
        self.open_percentile = open_percentile
        self.closed_percentile = closed_percentile
        self.closed_fraction = closed_fraction
        self.open_fraction = open_fraction
        self.min_range = min_range
        self.min_samples = min_samples
        self.threshold_min = threshold_min
        self.threshold_max = threshold_max

        self.start()

    def start(self):
        """Discard the learned baseline and calibrate again."""

        self.calibrated = False
        self.started_at = None
        self.samples = []

        self.threshold = self.default_threshold
        self.confidence = self.CONFIDENCE_DEFAULT
        self.open_ear = None
        self.closed_ear = None
        self.method = "default"

    def add(self, ear, timestamp):
        """
        Feed one frame's EAR. Returns the threshold to use for this frame
        (the default until calibration completes).
        """

        if self.calibrated:
            return self.threshold

        if self.started_at is None:
            self.started_at = timestamp

        self.samples.append(ear)

        if timestamp - self.started_at >= self.duration:
            self._finish()

        return self.threshold

    def _finish(self):

        if len(self.samples) < self.min_samples:
            # Too little data to say anything; keep the default and let
            # the risk engine know the eye signal is weak.
            self.calibrated = True
            return

        samples = np.asarray(self.samples, dtype=np.float64)

        # A percentile rather than the maximum: the driver blinks during
        # calibration, and a single noisy frame should not set the
        # baseline.
        self.open_ear = float(np.percentile(samples, self.open_percentile))
        closed = float(np.percentile(samples, self.closed_percentile))

        if self.open_ear - closed >= self.min_range:
            # Both ends seen, so the threshold can sit inside this
            # driver's own range and normalize for its width as well as
            # its offset.
            self.closed_ear = closed
            threshold = closed + self.closed_fraction * (
                self.open_ear - closed
            )
            self.confidence = self.CONFIDENCE_TWO_ENDED
            self.method = "two-ended"

        else:
            # No clear closure in the window: fall back to a fraction of
            # the open-eye baseline, which still removes the offset.
            threshold = self.open_fraction * self.open_ear
            self.confidence = self.CONFIDENCE_OPEN_ONLY
            self.method = "open-only"

        clamped = min(max(threshold, self.threshold_min), self.threshold_max)

        if clamped != threshold:
            # The learned value was implausible; it was clamped, so trust
            # the eye signal less.
            self.confidence = self.CONFIDENCE_OPEN_ONLY

        self.threshold = clamped
        self.calibrated = True

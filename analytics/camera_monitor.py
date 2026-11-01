"""
Telling a blinded camera apart from an empty seat.

Both leave the pipeline with no face, and before this they produced the
same verdict: DRIVER NOT VISIBLE. They are different problems. An empty
seat is the driver gone. A covered lens is the monitor defeated, which
in a fleet is a tamper event rather than a safety one, and it is also
what happens when a laptop's privacy shutter is left closed.

Brightness alone cannot separate them, because a dark cabin is dark
too. Variance can: a dark cabin still contains shapes, highlights and
sensor noise, so its pixels differ from each other. A covered lens is
flat. Both conditions have to hold for CAMERA_BLOCKED_TIME, so a camera
that has not finished waking up is not accused of being covered.

A dark flat frame is only one of the ways a feed goes blind, and on
this project's own test recording it was not the one that happened.
Covering the lens with a hand produced two thirds of a second of black
and then a still graphic, because Windows had noticed the camera was
obscured and substituted its own picture. Measured on that clip: 200
consecutive frames, bit identical, bright and full of detail. Every
brightness-and-variance test passed and the monitor reported nothing.

So a frozen feed is the second way in. Consecutive frames that are
identical to the bit never come from a live sensor -- noise alone
guarantees a difference -- so they mean the pipeline is being handed
the same picture over and over: a substituted graphic, a stalled
driver, a dropped stream. Either way the monitor cannot see, which is
the thing it exists to say.
"""

from app.config import (
    CAMERA_BLOCKED_BRIGHTNESS,
    CAMERA_BLOCKED_VARIANCE,
    CAMERA_BLOCKED_TIME,
    CAMERA_FROZEN_TIME
)


class CameraMonitor:

    def __init__(self, max_brightness=CAMERA_BLOCKED_BRIGHTNESS,
                 max_variance=CAMERA_BLOCKED_VARIANCE,
                 min_duration=CAMERA_BLOCKED_TIME,
                 frozen_duration=CAMERA_FROZEN_TIME):

        self.max_brightness = max_brightness
        self.max_variance = max_variance
        self.min_duration = min_duration
        self.frozen_duration = frozen_duration

        self.dark_since = None
        self.blocked = False
        self.frozen = False

        # The last frame's fingerprint, and when the current run of
        # identical ones began.
        self.fingerprint = None
        self.identical_since = None

        # One spell of a covered lens counts once, however long
        # it lasts.
        self.blocked_count = 0

    def update(self, brightness, variance, timestamp, fingerprint=None):
        """
        brightness:  mean pixel value of the frame, 0-255
        variance:    standard deviation of the same pixels
        fingerprint: a value that differs whenever the frame's pixels
                     do -- the pipeline passes a hash of the sample it
                     already took to measure the light. None skips the
                     frozen-feed check, which is what callers that
                     cannot cheaply fingerprint a frame get.
        """

        flat = (
            brightness <= self.max_brightness
            and variance <= self.max_variance
        )

        if flat:
            if self.dark_since is None:
                self.dark_since = timestamp

            dark_for = timestamp - self.dark_since

        else:
            self.dark_since = None
            dark_for = 0.0

        frozen_for = self._frozen_for(fingerprint, timestamp)

        self.frozen = frozen_for >= self.frozen_duration
        blocked = dark_for >= self.min_duration or self.frozen

        if blocked and not self.blocked:
            self.blocked_count += 1

        self.blocked = blocked
        duration = max(dark_for, frozen_for) if blocked else 0.0

        return {
            "blocked": self.blocked,
            "blocked_count": self.blocked_count,
            "brightness": brightness,
            "variance": variance,
            "blocked_duration": duration,
            "frozen": self.frozen,

            # What the dashboard tells the driver. A covered lens and a
            # substituted graphic need different words, because only one
            # of them is fixed by moving a hand.
            "reason": "frozen" if self.frozen else "flat"
        }

    def _frozen_for(self, fingerprint, timestamp):
        """
        How long the frame has been identical to the one before it.

        A single repeated frame is not evidence: some encoders pad with
        duplicates, and a dropped frame can be served twice. A whole
        second of them is, which is why this has its own threshold
        rather than borrowing the one for a dark lens.
        """

        if fingerprint is None:
            self.fingerprint = None
            self.identical_since = None

            return 0.0

        if fingerprint != self.fingerprint:
            self.fingerprint = fingerprint
            self.identical_since = timestamp

            return 0.0

        return timestamp - self.identical_since

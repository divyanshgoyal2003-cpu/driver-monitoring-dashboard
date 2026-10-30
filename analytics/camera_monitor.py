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
"""

from app.config import (
    CAMERA_BLOCKED_BRIGHTNESS,
    CAMERA_BLOCKED_VARIANCE,
    CAMERA_BLOCKED_TIME
)


class CameraMonitor:

    def __init__(self, max_brightness=CAMERA_BLOCKED_BRIGHTNESS,
                 max_variance=CAMERA_BLOCKED_VARIANCE,
                 min_duration=CAMERA_BLOCKED_TIME):

        self.max_brightness = max_brightness
        self.max_variance = max_variance
        self.min_duration = min_duration

        self.dark_since = None
        self.blocked = False

    def update(self, brightness, variance, timestamp):
        """
        brightness: mean pixel value of the frame, 0-255
        variance:   standard deviation of the same pixels
        """

        flat = (
            brightness <= self.max_brightness
            and variance <= self.max_variance
        )

        if flat:
            if self.dark_since is None:
                self.dark_since = timestamp

            duration = timestamp - self.dark_since

        else:
            self.dark_since = None
            duration = 0.0

        self.blocked = duration >= self.min_duration

        return {
            "blocked": self.blocked,
            "brightness": brightness,
            "variance": variance,
            "blocked_duration": duration if self.blocked else 0.0
        }

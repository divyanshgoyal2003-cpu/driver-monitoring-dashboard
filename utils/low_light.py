"""
Making the most of a dim cabin.

This is not night vision and cannot become it. Infrared needs an
illuminator and a sensor without an IR-cut filter; a laptop webcam has
neither, and no amount of processing recovers photons that were never
captured. What this does is redistribute the contrast that *is* there,
which can bring a face in a dim room back within MediaPipe's reach. It
does nothing for a cabin that is genuinely dark.

CLAHE is applied to the luma channel only, so colour survives. The
strength ramps with how dim the frame is, rather than switching on at
a threshold, so a frame hovering at the boundary does not flicker
between enhanced and raw.

Enhancement deliberately does not change the brightness the risk engine
reads. Making an image easier to look at does not make the measurement
more trustworthy, so the confidence layer must keep seeing the real
light level.
"""

import cv2

from app.config import (
    LOW_LIGHT_BRIGHTNESS,
    LOW_LIGHT_CLIP_LIMIT,
    LOW_LIGHT_GRID
)


class LowLightEnhancer:

    # Below this fraction of the threshold the full clip limit applies.
    FULL_STRENGTH_AT = 0.4

    def __init__(self, threshold=LOW_LIGHT_BRIGHTNESS,
                 clip_limit=LOW_LIGHT_CLIP_LIMIT,
                 grid=LOW_LIGHT_GRID):

        self.threshold = threshold
        self.clip_limit = clip_limit
        self.grid = grid

        # Rebuilt whenever the strength changes materially; CLAHE
        # objects are cheap to make but not free.
        self._clahe = None
        self._clahe_limit = None

    def strength(self, brightness):
        """0 in a well-lit frame, 1 in a dim one."""

        if brightness is None or brightness >= self.threshold:
            return 0.0

        floor = self.threshold * self.FULL_STRENGTH_AT

        if brightness <= floor:
            return 1.0

        return (self.threshold - brightness) / (self.threshold - floor)

    def enhance(self, frame, brightness):
        """
        A lifted copy of `frame`, or the frame itself when there is
        nothing to gain. The caller keeps the original brightness.
        """

        strength = self.strength(brightness)

        if strength <= 0.0:
            return frame

        limit = 1.0 + (self.clip_limit - 1.0) * strength

        if self._clahe is None or abs(limit - self._clahe_limit) > 0.25:
            self._clahe = cv2.createCLAHE(
                clipLimit=limit,
                tileGridSize=(self.grid, self.grid)
            )
            self._clahe_limit = limit

        # LAB rather than grayscale: the models want a colour frame,
        # and equalising L leaves the colour channels alone.
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        lightness, a, b = cv2.split(lab)

        lab = cv2.merge([self._clahe.apply(lightness), a, b])

        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

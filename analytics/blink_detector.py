from collections import deque

from scipy.spatial import distance

from analytics.ear_calibrator import EarCalibrator
from app.config import (
    EAR_THRESHOLD,
    ADAPTIVE_EAR,
    BLINK_MIN_DURATION,
    BLINK_MAX_DURATION,
    BLINK_RATE_WINDOW
)


class BlinkDetector:

    LEFT_EYE = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE = [362, 385, 387, 263, 373, 380]

    def __init__(self, ear_threshold=EAR_THRESHOLD,
                 min_duration=BLINK_MIN_DURATION,
                 max_duration=BLINK_MAX_DURATION,
                 rate_window=BLINK_RATE_WINDOW,
                 adaptive=ADAPTIVE_EAR):

        self.blink_count = 0
        self.closed_since = None
        self.blink_times = deque()
        self.start_time = None

        self.EAR_THRESHOLD = ear_threshold
        self.MIN_DURATION = min_duration
        self.MAX_DURATION = max_duration
        self.RATE_WINDOW = rate_window

        # The threshold is learned from this driver rather than fixed;
        # until calibration finishes it returns EAR_THRESHOLD.
        self.calibrator = (
            EarCalibrator(default_threshold=ear_threshold)
            if adaptive
            else None
        )

    def recalibrate(self):
        """Learn this driver's EAR baseline again (the C key)."""

        if self.calibrator is not None:
            self.calibrator.start()

    def eye_aspect_ratio(self, eye):
        A = distance.euclidean(eye[1], eye[5])
        B = distance.euclidean(eye[2], eye[4])
        C = distance.euclidean(eye[0], eye[3])

        return (A + B) / (2.0 * C)

    def update(self, landmarks, width, height, timestamp):

        if not landmarks:
            # A closure cannot be timed while the face is out of view, so
            # a half-finished blink is dropped instead of being counted
            # when the face comes back.
            self.closed_since = None
            return None

        if self.start_time is None:
            self.start_time = timestamp

        face = landmarks[0]

        left_eye = []
        right_eye = []

        for idx in self.LEFT_EYE:
            p = face[idx]
            left_eye.append((p.x * width, p.y * height))

        for idx in self.RIGHT_EYE:
            p = face[idx]
            right_eye.append((p.x * width, p.y * height))

        left_ear = self.eye_aspect_ratio(left_eye)
        right_ear = self.eye_aspect_ratio(right_eye)

        ear = (left_ear + right_ear) / 2

        if self.calibrator is not None:
            threshold = self.calibrator.add(ear, timestamp)
            confidence = self.calibrator.confidence
        else:
            threshold = self.EAR_THRESHOLD
            confidence = 1.0

        eyes_closed = ear < threshold

        if eyes_closed:
            if self.closed_since is None:
                self.closed_since = timestamp

        elif self.closed_since is not None:
            duration = timestamp - self.closed_since

            if self.MIN_DURATION <= duration < self.MAX_DURATION:
                self.blink_count += 1
                self.blink_times.append(timestamp)

            self.closed_since = None

        while (self.blink_times
               and timestamp - self.blink_times[0] > self.RATE_WINDOW):
            self.blink_times.popleft()

        closed_duration = (
            timestamp - self.closed_since
            if self.closed_since is not None
            else 0.0
        )

        return {
            "ear": ear,
            "blink_count": self.blink_count,
            "blink_rate": self._blink_rate(timestamp),
            "eyes_closed": eyes_closed,
            "closed_duration": closed_duration,
            "threshold": threshold,
            "calibrated": (
                self.calibrator.calibrated
                if self.calibrator is not None
                else True
            ),
            "eye_confidence": confidence
        }

    def _blink_rate(self, timestamp):
        """Blinks per minute over the last RATE_WINDOW seconds."""

        span = min(self.RATE_WINDOW, timestamp - self.start_time)

        if span <= 0:
            return 0.0

        return len(self.blink_times) * 60.0 / span

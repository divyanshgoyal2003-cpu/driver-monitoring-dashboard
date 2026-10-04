from collections import deque

from scipy.spatial import distance

from app.config import MAR_THRESHOLD, YAWN_MIN_DURATION, YAWN_RATE_WINDOW


class YawnDetector:

    # MediaPipe mouth landmarks
    MOUTH_LEFT = 61
    MOUTH_RIGHT = 291

    MOUTH_TOP = 13
    MOUTH_BOTTOM = 14

    def __init__(self, mar_threshold=MAR_THRESHOLD,
                 min_duration=YAWN_MIN_DURATION,
                 rate_window=YAWN_RATE_WINDOW):

        self.yawn_count = 0
        self.open_since = None
        self.yawn_times = deque()

        # See app/config.py for how the MAR threshold was derived.
        self.MAR_THRESHOLD = mar_threshold
        self.MIN_DURATION = min_duration
        self.RATE_WINDOW = rate_window

    def mouth_aspect_ratio(self, face, width, height):

        horizontal = distance.euclidean(
            (face[self.MOUTH_LEFT].x * width,
             face[self.MOUTH_LEFT].y * height),
            (face[self.MOUTH_RIGHT].x * width,
             face[self.MOUTH_RIGHT].y * height)
        )

        vertical = distance.euclidean(
            (face[self.MOUTH_TOP].x * width,
             face[self.MOUTH_TOP].y * height),
            (face[self.MOUTH_BOTTOM].x * width,
             face[self.MOUTH_BOTTOM].y * height)
        )

        if horizontal == 0:
            return 0.0

        return vertical / horizontal

    def update(self, landmarks, width, height, timestamp):

        if not landmarks:
            # Drop a half-finished yawn rather than counting it when the
            # face reappears with the mouth closed.
            self.open_since = None
            return None

        face = landmarks[0]

        mar = self.mouth_aspect_ratio(face, width, height)

        mouth_open = mar > self.MAR_THRESHOLD

        if mouth_open:
            if self.open_since is None:
                self.open_since = timestamp

        elif self.open_since is not None:
            if timestamp - self.open_since >= self.MIN_DURATION:
                self.yawn_count += 1
                self.yawn_times.append(timestamp)

            self.open_since = None

        while (self.yawn_times
               and timestamp - self.yawn_times[0] > self.RATE_WINDOW):
            self.yawn_times.popleft()

        open_duration = (
            timestamp - self.open_since
            if self.open_since is not None
            else 0.0
        )

        return {
            "mar": mar,
            "mouth_open": mouth_open,
            "open_duration": open_duration,
            "yawn_count": self.yawn_count,
            "recent_yawns": len(self.yawn_times)
        }

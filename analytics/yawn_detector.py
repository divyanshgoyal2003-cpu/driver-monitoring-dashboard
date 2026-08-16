from scipy.spatial import distance


class YawnDetector:

    # MediaPipe mouth landmarks
    MOUTH_LEFT = 61
    MOUTH_RIGHT = 291

    MOUTH_TOP = 13
    MOUTH_BOTTOM = 14

    def __init__(self):

        self.yawn_count = 0
        self.open_frames = 0

        self.MAR_THRESHOLD = 0.6
        self.MIN_FRAMES = 8

    def mouth_aspect_ratio(self, face):

        horizontal = distance.euclidean(
            (face[self.MOUTH_LEFT].x, face[self.MOUTH_LEFT].y),
            (face[self.MOUTH_RIGHT].x, face[self.MOUTH_RIGHT].y)
        )

        vertical = distance.euclidean(
            (face[self.MOUTH_TOP].x, face[self.MOUTH_TOP].y),
            (face[self.MOUTH_BOTTOM].x, face[self.MOUTH_BOTTOM].y)
        )

        if horizontal == 0:
            return 0.0

        return vertical / horizontal

    def update(self, landmarks):

        if not landmarks:
            return None

        face = landmarks[0]

        mar = self.mouth_aspect_ratio(face)

        mouth_open = mar > self.MAR_THRESHOLD

        if mouth_open:
            self.open_frames += 1

        else:
            if self.open_frames >= self.MIN_FRAMES:
                self.yawn_count += 1

            self.open_frames = 0

        return {
            "mar": mar,
            "mouth_open": mouth_open,
            "yawn_count": self.yawn_count
        }
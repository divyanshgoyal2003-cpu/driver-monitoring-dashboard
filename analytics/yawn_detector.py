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

        # Tuned against pixel-space MAR. The previous value of 0.6 was
        # tuned against normalized coordinates, where x and y are scaled
        # by different amounts and the ratio therefore drifts with head
        # roll. 0.6 * (720 / 1280) = 0.3375 preserves the old behaviour
        # for a level head at 720p while staying correct at any
        # resolution and any roll angle.
        self.MAR_THRESHOLD = 0.34
        self.MIN_FRAMES = 8

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

    def update(self, landmarks, width, height):

        if not landmarks:
            return None

        face = landmarks[0]

        mar = self.mouth_aspect_ratio(face, width, height)

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

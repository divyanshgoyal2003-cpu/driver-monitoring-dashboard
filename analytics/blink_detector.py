from scipy.spatial import distance


class BlinkDetector:

    LEFT_EYE = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE = [362, 385, 387, 263, 373, 380]

    def __init__(self):
        self.blink_count = 0
        self.closed_frames = 0

        self.EAR_THRESHOLD = 0.23
        self.MIN_FRAMES = 2

    def eye_aspect_ratio(self, eye):
        A = distance.euclidean(eye[1], eye[5])
        B = distance.euclidean(eye[2], eye[4])
        C = distance.euclidean(eye[0], eye[3])

        return (A + B) / (2.0 * C)

    def update(self, landmarks, width, height):

        if not landmarks:
            return None

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

        if ear < self.EAR_THRESHOLD:
            self.closed_frames += 1
        else:
            if self.closed_frames >= self.MIN_FRAMES:
                self.blink_count += 1
            self.closed_frames = 0

        return {
            "ear": ear,
            "blink_count": self.blink_count,
            "eyes_closed": ear < self.EAR_THRESHOLD
        }
import cv2


class LandmarkRenderer:

    def draw(self, frame, face_landmarks):

        if not face_landmarks:
            return frame

        h, w, _ = frame.shape

        for landmarks in face_landmarks:

            for landmark in landmarks:

                x = int(landmark.x * w)
                y = int(landmark.y * h)

                cv2.circle(
                    frame,
                    (x, y),
                    1,
                    (0, 255, 255),
                    -1
                )

        return frame
import cv2
import time

from models.detectors.yolo_face_detector import YOLOFaceDetector


class VideoProcessor:
    def __init__(self):
        self.prev_time = time.time()
        self.face_detector = YOLOFaceDetector()

    def process(self, frame):

        current_time = time.time()

        fps = 1 / (current_time - self.prev_time)

        self.prev_time = current_time

        frame, faces = self.face_detector.track(frame)

        cv2.putText(
            frame,
            f"FPS : {fps:.2f}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Faces : {len(faces)}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2
        )

        return frame
import cv2
import time

from models.detectors.yolo_face_detector import YOLOFaceDetector
from app.config import FACE_LANDMARK_MODEL
from models.landmarks.mediapipe_face_landmarker import MediaPipeFaceLandmarker
from analytics.blink_detector import BlinkDetector
from renderers.landmark_renderer import LandmarkRenderer
from renderers.dashboard_renderer import DashboardRenderer

class VideoProcessor:
    def __init__(self):
        self.prev_time = time.time()
        self.face_detector = YOLOFaceDetector()
        self.landmarker = MediaPipeFaceLandmarker(FACE_LANDMARK_MODEL)
        self.blink_detector = BlinkDetector()
        self.landmark_renderer = LandmarkRenderer()
        self.dashboard_renderer = DashboardRenderer()

    def process(self, frame):

        current_time = time.time()

        fps = 1 / (current_time - self.prev_time)

        self.prev_time = current_time

        frame, faces = self.face_detector.track(frame)
        result = self.landmarker.detect(frame)

        h, w, _ = frame.shape

        blink_data = self.blink_detector.update(
            result.face_landmarks,
            w,
            h
        )
        if result.face_landmarks:

            frame = self.landmark_renderer.draw(
                 frame,
                 result.face_landmarks
            )

        frame = self.dashboard_renderer.draw(
            frame,
            fps,
            len(faces),
             blink_data
        )
        return frame
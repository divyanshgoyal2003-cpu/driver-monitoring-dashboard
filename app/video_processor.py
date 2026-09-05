import cv2
import time

from models.detectors.yolo_face_detector import YOLOFaceDetector
from app.config import FACE_LANDMARK_MODEL
from models.landmarks.mediapipe_face_landmarker import MediaPipeFaceLandmarker
from analytics.blink_detector import BlinkDetector
from analytics.yawn_detector import YawnDetector
from analytics.head_pose_estimator import HeadPoseEstimator
from renderers.landmark_renderer import LandmarkRenderer
from renderers.dashboard_renderer import DashboardRenderer


class VideoProcessor:
    def __init__(self):
        self.prev_time = time.time()
        self.face_detector = YOLOFaceDetector()
        self.landmarker = MediaPipeFaceLandmarker(FACE_LANDMARK_MODEL)
        self.blink_detector = BlinkDetector()
        self.yawn_detector = YawnDetector()
        self.head_pose_estimator = None
        self.landmark_renderer = LandmarkRenderer()
        self.dashboard_renderer = DashboardRenderer()

    def process(self, frame):

        current_time = time.time()

        elapsed = current_time - self.prev_time
        fps = 1 / elapsed if elapsed > 0 else 0.0

        self.prev_time = current_time

        h, w, _ = frame.shape

        # Landmarks are detected on the untouched frame. track() returns
        # a copy with boxes and ID labels drawn on it, which the
        # landmarker should not see.
        result = self.landmarker.detect(frame)

        annotated, faces = self.face_detector.track(frame)

        if self.head_pose_estimator is None:
            self.head_pose_estimator = HeadPoseEstimator(w, h)

        blink_data = self.blink_detector.update(
            result.face_landmarks,
            w,
            h
        )
        yawn_data = self.yawn_detector.update(
            result.face_landmarks,
            w,
            h
        )
        head_pose_data = self.head_pose_estimator.estimate(
            result.face_landmarks
        )

        if result.face_landmarks:

            annotated = self.landmark_renderer.draw(
                annotated,
                result.face_landmarks
            )

        annotated = self.dashboard_renderer.draw(
            annotated,
            fps,
            len(faces),
            blink_data,
            yawn_data,
            head_pose_data
        )

        return annotated

import time

from app.config import FACE_LANDMARK_MODEL, SHOW_LANDMARKS
from models.detectors.yolo_face_detector import YOLOFaceDetector
from models.landmarks.mediapipe_face_landmarker import MediaPipeFaceLandmarker
from analytics.blink_detector import BlinkDetector
from analytics.yawn_detector import YawnDetector
from analytics.head_pose_estimator import HeadPoseEstimator
from analytics.gaze_estimator import GazeEstimator
from analytics.drowsiness_detector import DrowsinessDetector
from analytics.distraction_detector import DistractionDetector
from renderers.landmark_renderer import LandmarkRenderer
from renderers.overlay_renderer import OverlayRenderer
from renderers.dashboard_renderer import DashboardRenderer


class VideoProcessor:

    NOSE = 1

    def __init__(self):

        self.face_detector = YOLOFaceDetector()
        self.landmarker = MediaPipeFaceLandmarker(FACE_LANDMARK_MODEL)
        self.blink_detector = BlinkDetector()
        self.yawn_detector = YawnDetector()
        self.head_pose_estimator = None
        self.gaze_estimator = GazeEstimator()
        self.drowsiness_detector = DrowsinessDetector()
        self.distraction_detector = DistractionDetector()
        self.landmark_renderer = LandmarkRenderer()
        self.overlay_renderer = OverlayRenderer()
        self.dashboard_renderer = DashboardRenderer()

        self.show_landmarks = SHOW_LANDMARKS

        self.prev_time = None
        self.fps = 0.0

        # Everything measured on the latest frame, for rendering and
        # logging.
        self.state = {}

    def process(self, frame, timestamp=None):
        """
        timestamp is in seconds and drives every time-based threshold.
        It defaults to wall-clock time; pass the video position when
        processing a file.
        """

        if timestamp is None:
            timestamp = time.monotonic()

        self._update_fps()

        h, w, _ = frame.shape

        # Landmarks are detected on the untouched frame; overlays are
        # only drawn afterwards, on a copy.
        result = self.landmarker.detect(frame, timestamp)
        landmarks = result.face_landmarks

        faces = self.face_detector.track(frame)

        if not landmarks and faces:
            # The face is too small for MediaPipe's own detector; retry
            # inside the largest YOLO box.
            largest = max(faces, key=self._box_area)
            landmarks = self.landmarker.detect_in_region(
                frame,
                largest["bbox"]
            )

        driver_id = self._mark_driver(faces, landmarks, w, h)

        if self.head_pose_estimator is None:
            self.head_pose_estimator = HeadPoseEstimator(w, h)

        blink_data = self.blink_detector.update(
            landmarks,
            w,
            h,
            timestamp
        )
        yawn_data = self.yawn_detector.update(
            landmarks,
            w,
            h,
            timestamp
        )
        head_pose_data = self.head_pose_estimator.estimate(
            landmarks,
            timestamp
        )
        gaze_data = self.gaze_estimator.estimate(
            landmarks,
            w,
            h,
            head_pose_data
        )
        drowsiness_data = self.drowsiness_detector.update(
            blink_data,
            timestamp
        )
        distraction_data = self.distraction_detector.update(
            head_pose_data,
            gaze_data,
            timestamp
        )

        self.state = {
            "timestamp": timestamp,
            "fps": self.fps,
            "faces": faces,
            "face_count": len(faces),
            "driver_id": driver_id,
            "face_detected": bool(landmarks),
            "landmarks": landmarks,
            "blink": blink_data,
            "drowsiness": drowsiness_data,
            "yawn": yawn_data,
            "head_pose": head_pose_data,
            "gaze": gaze_data,
            "distraction": distraction_data
        }

        annotated = frame.copy()

        if self.show_landmarks and landmarks:
            annotated = self.landmark_renderer.draw(annotated, landmarks)

        annotated = self.overlay_renderer.draw(annotated, self.state)

        return self.dashboard_renderer.draw(annotated, self.state)

    def recalibrate(self):

        if self.head_pose_estimator is not None:
            self.head_pose_estimator.start_calibration()

    def close(self):
        self.landmarker.close()

    def _update_fps(self):

        now = time.perf_counter()

        if self.prev_time is not None:
            elapsed = now - self.prev_time

            if elapsed > 0:
                instant = 1.0 / elapsed

                # Exponential smoothing keeps the reading steady.
                if self.fps == 0.0:
                    self.fps = instant
                else:
                    self.fps = 0.9 * self.fps + 0.1 * instant

        self.prev_time = now

    @staticmethod
    def _box_area(face):

        x1, y1, x2, y2 = face["bbox"]

        return (x2 - x1) * (y2 - y1)

    def _mark_driver(self, faces, landmarks, width, height):
        """
        The analysed face is the one MediaPipe returned. Mark the YOLO
        box that contains its nose as the driver and return that box's
        track ID.
        """

        if not landmarks:
            return None

        nose = landmarks[0][self.NOSE]
        nx, ny = nose.x * width, nose.y * height

        for face in faces:
            x1, y1, x2, y2 = face["bbox"]

            if x1 <= nx <= x2 and y1 <= ny <= y2:
                face["is_driver"] = True
                return face["id"]

        return None

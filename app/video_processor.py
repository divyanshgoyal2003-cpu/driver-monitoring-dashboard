import time

import cv2

from app.config import (
    FACE_LANDMARK_MODEL,
    OPENCV_THREADS,
    SHOW_LANDMARKS,
    PHONE_DETECTION_INTERVAL,
    EMOTION_INTERVAL,
    SEATBELT_INTERVAL,
    DYNAMIC_WEIGHTS,
    LOW_LIGHT_ENHANCE
)
from models.detectors.yolo_face_detector import YOLOFaceDetector
from models.detectors.yolo_object_detector import YOLOObjectDetector
from models.detectors.seatbelt_detector import SeatbeltDetector
from models.landmarks.mediapipe_face_landmarker import MediaPipeFaceLandmarker
from models.classifiers.emotion_classifier import EmotionClassifier
from analytics.blink_detector import BlinkDetector
from analytics.yawn_detector import YawnDetector
from analytics.head_pose_estimator import HeadPoseEstimator
from analytics.gaze_estimator import GazeEstimator
from analytics.drowsiness_detector import DrowsinessDetector
from analytics.distraction_detector import DistractionDetector
from analytics.phone_usage_detector import PhoneUsageDetector
from analytics.emotion_analyzer import EmotionAnalyzer
from analytics.seatbelt_monitor import SeatbeltMonitor
from analytics.camera_monitor import CameraMonitor
from risk_engine.scoring import DriverSafetyIndex
from risk_engine.context import ContextProvider
from utils.low_light import LowLightEnhancer
from renderers.landmark_renderer import LandmarkRenderer
from renderers.overlay_renderer import OverlayRenderer
from renderers.dashboard_renderer import DashboardRenderer


class VideoProcessor:

    NOSE = 1

    # The frame is sampled every Nth pixel to read its brightness:
    # exact enough for a 0-255 scale, and effectively free.
    BRIGHTNESS_STRIDE = 8

    def __init__(self, dynamic_weights=DYNAMIC_WEIGHTS,
                 context_provider=None, enhance=LOW_LIGHT_ENHANCE):

        self.context_provider = context_provider or ContextProvider()
        self.enhancer = LowLightEnhancer() if enhance else None

        self.face_detector = YOLOFaceDetector()
        self.landmarker = MediaPipeFaceLandmarker(FACE_LANDMARK_MODEL)
        self.phone_detector = YOLOObjectDetector()
        self.emotion_classifier = EmotionClassifier()
        self.seatbelt_detector = SeatbeltDetector()
        self.blink_detector = BlinkDetector()
        self.yawn_detector = YawnDetector()
        self.head_pose_estimator = None
        self.gaze_estimator = GazeEstimator()
        self.drowsiness_detector = DrowsinessDetector()
        self.distraction_detector = DistractionDetector()
        self.phone_usage_detector = PhoneUsageDetector()
        self.emotion_analyzer = EmotionAnalyzer(EmotionClassifier.LABELS)
        self.seatbelt_monitor = SeatbeltMonitor()
        self.camera_monitor = CameraMonitor()
        self.safety_index = DriverSafetyIndex(dynamic=dynamic_weights)
        self.landmark_renderer = LandmarkRenderer()
        self.overlay_renderer = OverlayRenderer()
        self.dashboard_renderer = DashboardRenderer()

        self.show_landmarks = SHOW_LANDMARKS

        # ultralytics turned OpenCV threading off when it was imported.
        cv2.setNumThreads(OPENCV_THREADS)

        self.prev_time = None
        self.fps = 0.0

        self.frame_index = 0
        self.phones = []

        # Everything measured on the latest frame, for rendering and
        # logging.
        self.state = {}

    def process(self, frame, timestamp=None, render=True):
        """
        timestamp is in seconds and drives every time-based threshold.
        It defaults to wall-clock time; pass the video position when
        processing a file.

        render=False fills in self.state and returns None, skipping the
        overlays. Offline analysis (see the evaluation package) does not
        need a picture, and drawing one is most of the cheap frames.
        """

        if timestamp is None:
            timestamp = time.monotonic()

        self._update_fps()

        h, w, _ = frame.shape

        # Read the light before anything touches the frame. brightness
        # is what the confidence layer and the context layer are given,
        # so it has to describe the camera's real view, not the lifted
        # copy the models go on to see.
        brightness, variance = self._luminance(frame)

        camera_data = self.camera_monitor.update(
            brightness,
            variance,
            timestamp
        )

        enhancement = 0.0

        if self.enhancer is not None:
            enhancement = self.enhancer.strength(brightness)
            frame = self.enhancer.enhance(frame, brightness)

        # Landmarks are detected on the unannotated frame; overlays are
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

        driver = self._mark_driver(faces, landmarks, w, h)
        driver_box = self._driver_box(driver, landmarks, w, h)

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

        # Phone detection runs on every Nth frame; the last result is
        # reused in between.
        if self.frame_index % PHONE_DETECTION_INTERVAL == 0:
            self.phones = self.phone_detector.detect(frame)

        phone_data = self.phone_usage_detector.update(
            self.phones,
            timestamp
        )

        # Staggered from phone detection so the two models do not run on
        # the same frame.
        emotion_probabilities = None

        if (driver_box is not None
                and self.frame_index % EMOTION_INTERVAL
                == EMOTION_INTERVAL // 2):
            emotion_probabilities = self.emotion_classifier.predict(
                frame,
                driver_box
            )

        emotion_data = self.emotion_analyzer.update(
            emotion_probabilities,
            bool(landmarks)
        )

        seatbelt_observation = None

        if self.frame_index % SEATBELT_INTERVAL == SEATBELT_INTERVAL - 1:
            seatbelt_observation = self.seatbelt_detector.detect(
                frame,
                driver_box
            )

        seatbelt_data = self.seatbelt_monitor.update(
            seatbelt_observation,
            timestamp
        )

        self.frame_index += 1

        context = self.context_provider.update(timestamp, brightness)

        self.state = {
            "timestamp": timestamp,
            "fps": self.fps,
            "brightness": brightness,
            "camera": camera_data,
            "enhancement": enhancement,
            "faces": faces,
            "face_count": len(faces),
            "driver_id": driver["id"] if driver else None,
            "driver_box": driver_box,
            "face_detected": bool(landmarks),
            "landmarks": landmarks,
            "blink": blink_data,
            "drowsiness": drowsiness_data,
            "yawn": yawn_data,
            "head_pose": head_pose_data,
            "gaze": gaze_data,
            "distraction": distraction_data,
            "phone": phone_data,
            "emotion": emotion_data,
            "seatbelt": seatbelt_data
        }

        self.state["risk"] = self.safety_index.update(
            self.state, timestamp, context
        )

        if not render:
            return None

        annotated = frame.copy()

        if self.show_landmarks and landmarks:
            annotated = self.landmark_renderer.draw(annotated, landmarks)

        annotated = self.overlay_renderer.draw(annotated, self.state)

        return self.dashboard_renderer.draw(annotated, self.state)

    def recalibrate(self):
        """Learn the driver's neutral head pose and EAR range again."""

        if self.head_pose_estimator is not None:
            self.head_pose_estimator.start_calibration()

        self.blink_detector.recalibrate()

    def close(self):
        self.landmarker.close()

    def _luminance(self, frame):
        """
        (mean, standard deviation) of the frame's pixels, 0-255.

        The spread is what separates a covered lens from a dark cabin,
        so both come from the same sample.
        """

        stride = self.BRIGHTNESS_STRIDE
        sample = frame[::stride, ::stride]

        return float(sample.mean()), float(sample.std())

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
        box that contains its nose as the driver and return that face.
        """

        if not landmarks:
            return None

        nose = landmarks[0][self.NOSE]
        nx, ny = nose.x * width, nose.y * height

        for face in faces:
            x1, y1, x2, y2 = face["bbox"]

            if x1 <= nx <= x2 and y1 <= ny <= y2:
                face["is_driver"] = True
                return face

        return None

    @staticmethod
    def _driver_box(driver, landmarks, width, height):
        """
        The driver's face box: the YOLO box when there is one, otherwise
        the bounding box of the landmarks. None without a face.
        """

        if driver is not None:
            return driver["bbox"]

        if not landmarks:
            return None

        xs = [p.x * width for p in landmarks[0]]
        ys = [p.y * height for p in landmarks[0]]

        return (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))

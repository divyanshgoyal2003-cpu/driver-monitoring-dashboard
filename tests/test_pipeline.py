"""
End-to-end test of VideoProcessor with the real models.

Uses a sample photo that ships with ultralytics, cropped so the face is
close to the camera like a driver's would be. Skipped when the models
have not been downloaded (python download_models.py).
"""

import os
import unittest

import cv2
import numpy as np

from app import config

REQUIRED_MODELS = [
    config.YOLO_FACE_MODEL,
    config.FACE_LANDMARK_MODEL,
    config.OBJECT_DETECTION_MODEL,
]

MODELS_PRESENT = all(os.path.exists(path) for path in REQUIRED_MODELS)

# 16:9 region of ultralytics/assets/zidane.jpg around the right-hand
# face, sized so the face fills about half the frame height like a
# driver's face in front of a dashboard camera.
DRIVER_CROP = (569, 0, 1280, 400)


def sample_image():

    import ultralytics

    path = os.path.join(
        os.path.dirname(ultralytics.__file__), "assets", "zidane.jpg"
    )

    return cv2.imread(path)


def sample_driver_frame():

    x1, y1, x2, y2 = DRIVER_CROP

    return cv2.resize(sample_image()[y1:y2, x1:x2], (1280, 720))


@unittest.skipUnless(MODELS_PRESENT, "models not downloaded")
class PipelineTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        from app.video_processor import VideoProcessor

        cls.processor = VideoProcessor()
        cls.driver_frame = sample_driver_frame()
        cls.timestamp = 0.0

    @classmethod
    def tearDownClass(cls):
        cls.processor.close()

    def process(self, frame):

        output = self.processor.process(frame, self.timestamp)
        type(self).timestamp += 1 / 30

        return output, self.processor.state

    def test_driver_frame(self):

        for _ in range(5):
            output, state = self.process(self.driver_frame)

        expected_width = 1280 + config.PANEL_WIDTH
        self.assertEqual(output.shape, (720, expected_width, 3))

        self.assertTrue(state["face_detected"])
        self.assertGreaterEqual(state["face_count"], 1)
        self.assertIsNotNone(state["driver_id"])

        self.assertIsNotNone(state["blink"])
        self.assertIsNotNone(state["yawn"])

        head_pose = state["head_pose"]
        self.assertIsNotNone(head_pose)

        # The driver's nose points toward the right of the image, i.e.
        # the head is turned to the driver's left (negative yaw).
        self.assertLess(head_pose["yaw"], -5)

    def test_frame_without_face(self):

        blank = np.zeros((720, 1280, 3), dtype=np.uint8)

        output, state = self.process(blank)

        self.assertEqual(output.shape[0], config.DISPLAY_HEIGHT)
        self.assertFalse(state["face_detected"])
        self.assertIsNone(state["blink"])
        self.assertIsNone(state["head_pose"])

    def test_small_faces_use_region_fallback(self):

        # In the uncropped photo the faces are too small for MediaPipe's
        # own face detector; the YOLO-box fallback must still find one.
        image = sample_image()

        direct = self.processor.landmarker.region_landmarker.detect(
            __import__("mediapipe").Image(
                image_format=__import__("mediapipe").ImageFormat.SRGB,
                data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            )
        )
        self.assertEqual(len(direct.face_landmarks), 0)

        # ByteTrack only assigns an ID once a new face has been seen on
        # consecutive frames, so give it a few.
        for _ in range(3):
            _, state = self.process(image)

        self.assertTrue(state["face_detected"])
        self.assertIsNotNone(state["head_pose"])

        # The driver is the YOLO box containing the mapped nose
        # landmark, so this also checks the coordinate mapping.
        drivers = [face for face in state["faces"] if face["is_driver"]]
        self.assertEqual(len(drivers), 1)
        self.assertIsNotNone(state["driver_id"])

    def test_region_landmarks_map_into_the_box(self):

        image = sample_image()
        h, w = image.shape[:2]

        faces = self.processor.face_detector.detect(image)
        self.assertGreaterEqual(len(faces), 1)

        mapped = 0

        # One of the two faces is in near-profile, which MediaPipe does
        # not find at any crop size; only the frontal one must be found.
        for face in faces:
            x1, y1, x2, y2 = face["bbox"]

            found = self.processor.landmarker.detect_in_region(
                image, face["bbox"]
            )

            if not found:
                continue

            nose = found[0][1]
            self.assertTrue(x1 <= nose.x * w <= x2, face["bbox"])
            self.assertTrue(y1 <= nose.y * h <= y2, face["bbox"])
            mapped += 1

        self.assertGreaterEqual(mapped, 1)

    def test_no_phone_in_sample_frame(self):

        _, state = self.process(self.driver_frame)

        self.assertFalse(state["phone"]["detected"])
        self.assertFalse(state["phone"]["in_use"])

    def test_object_detector_finds_requested_class(self):

        # No phone photo ships with the libraries, so exercise the same
        # detection path with the "person" class on bus.jpg.
        import ultralytics
        from models.detectors.yolo_object_detector import YOLOObjectDetector

        bus = cv2.imread(os.path.join(
            os.path.dirname(ultralytics.__file__), "assets", "bus.jpg"
        ))

        detector = YOLOObjectDetector(class_names=("person",))
        people = detector.detect(bus)

        self.assertGreaterEqual(len(people), 3)
        self.assertTrue(all(p["label"] == "person" for p in people))

        phones = YOLOObjectDetector().class_ids
        self.assertEqual(len(phones), 1)

    def test_object_detector_rejects_unknown_class(self):

        from models.detectors.yolo_object_detector import YOLOObjectDetector

        with self.assertRaises(ValueError):
            YOLOObjectDetector(class_names=("seatbelt",))

    def test_other_resolution_is_scaled_for_display(self):

        small = cv2.resize(self.driver_frame, (640, 360))

        output, _ = self.process(small)

        self.assertEqual(output.shape[0], config.DISPLAY_HEIGHT)
        self.assertEqual(output.shape[1], 1280 + config.PANEL_WIDTH)


if __name__ == "__main__":
    unittest.main()

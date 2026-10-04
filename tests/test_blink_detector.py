import unittest

from analytics.blink_detector import BlinkDetector
from tests.fakes import make_face, feed

OPEN = make_face(ear=0.30)
CLOSED = make_face(ear=0.10)


class BlinkDetectorTest(unittest.TestCase):

    def setUp(self):
        self.detector = BlinkDetector()

    def test_ear_matches_geometry(self):
        result, _ = feed(self.detector, OPEN, 0.0, 0.1)
        self.assertAlmostEqual(result["ear"], 0.30, places=3)
        self.assertFalse(result["eyes_closed"])

    def test_normal_blink_is_counted(self):
        _, t = feed(self.detector, OPEN, 0.0, 1.0)
        _, t = feed(self.detector, CLOSED, t, 0.15)
        result, _ = feed(self.detector, OPEN, t, 0.5)
        self.assertEqual(result["blink_count"], 1)

    def test_single_frame_flicker_at_30fps_is_ignored(self):
        _, t = feed(self.detector, OPEN, 0.0, 1.0)
        _, t = feed(self.detector, CLOSED, t, 1 / 30)
        result, _ = feed(self.detector, OPEN, t, 0.5)
        self.assertEqual(result["blink_count"], 0)

    def test_blink_counted_at_low_frame_rate(self):
        # At 10 FPS a whole blink can fit in one frame.
        _, t = feed(self.detector, OPEN, 0.0, 1.0, fps=10)
        _, t = feed(self.detector, CLOSED, t, 0.1, fps=10)
        result, _ = feed(self.detector, OPEN, t, 0.5, fps=10)
        self.assertEqual(result["blink_count"], 1)

    def test_long_closure_is_not_a_blink(self):
        _, t = feed(self.detector, OPEN, 0.0, 1.0)
        during, t = feed(self.detector, CLOSED, t, 2.0)
        self.assertTrue(during["eyes_closed"])
        self.assertGreaterEqual(during["closed_duration"], 1.9)

        result, _ = feed(self.detector, OPEN, t, 0.5)
        self.assertEqual(result["blink_count"], 0)
        self.assertEqual(result["closed_duration"], 0.0)

    def test_face_lost_mid_blink_is_not_counted(self):
        _, t = feed(self.detector, OPEN, 0.0, 1.0)
        _, t = feed(self.detector, CLOSED, t, 0.1)

        lost, t = feed(self.detector, None, t, 0.5)
        self.assertIsNone(lost)

        result, _ = feed(self.detector, OPEN, t, 0.5)
        self.assertEqual(result["blink_count"], 0)

    def test_blink_rate_per_minute(self):
        t = 0.0

        for _ in range(10):
            _, t = feed(self.detector, OPEN, t, 5.8)
            _, t = feed(self.detector, CLOSED, t, 0.2)

        result, _ = feed(self.detector, OPEN, t, 0.1)

        self.assertEqual(result["blink_count"], 10)
        self.assertAlmostEqual(result["blink_rate"], 10.0, delta=0.5)


if __name__ == "__main__":
    unittest.main()

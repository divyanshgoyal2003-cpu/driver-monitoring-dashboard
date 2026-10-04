import unittest

from analytics.yawn_detector import YawnDetector
from tests.fakes import make_face, feed

CLOSED = make_face(mar=0.05)
WIDE_OPEN = make_face(mar=0.60)


class YawnDetectorTest(unittest.TestCase):

    def setUp(self):
        self.detector = YawnDetector()

    def test_mar_matches_geometry(self):
        result, _ = feed(self.detector, WIDE_OPEN, 0.0, 0.1)
        self.assertAlmostEqual(result["mar"], 0.60, places=3)
        self.assertTrue(result["mouth_open"])

    def test_yawn_counted_when_mouth_closes(self):
        _, t = feed(self.detector, CLOSED, 0.0, 1.0)
        during, t = feed(self.detector, WIDE_OPEN, t, 2.0)
        self.assertEqual(during["yawn_count"], 0)

        result, _ = feed(self.detector, CLOSED, t, 0.5)
        self.assertEqual(result["yawn_count"], 1)
        self.assertEqual(result["recent_yawns"], 1)

    def test_short_mouth_opening_is_not_a_yawn(self):
        _, t = feed(self.detector, CLOSED, 0.0, 1.0)
        _, t = feed(self.detector, WIDE_OPEN, t, 0.4)
        result, _ = feed(self.detector, CLOSED, t, 0.5)
        self.assertEqual(result["yawn_count"], 0)

    def test_face_lost_mid_yawn_is_not_counted(self):
        _, t = feed(self.detector, WIDE_OPEN, 0.0, 2.0)
        _, t = feed(self.detector, None, t, 0.5)
        result, _ = feed(self.detector, CLOSED, t, 0.5)
        self.assertEqual(result["yawn_count"], 0)

    def test_recent_yawns_expire(self):
        _, t = feed(self.detector, WIDE_OPEN, 0.0, 2.0)
        result, t = feed(self.detector, CLOSED, t, 1.0)
        self.assertEqual(result["recent_yawns"], 1)

        detector = self.detector
        later = detector.update([CLOSED], 1280, 720, t + 400.0)
        self.assertEqual(later["recent_yawns"], 0)
        self.assertEqual(later["yawn_count"], 1)


if __name__ == "__main__":
    unittest.main()

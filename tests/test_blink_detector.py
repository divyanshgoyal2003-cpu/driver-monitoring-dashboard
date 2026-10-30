import unittest

from analytics.blink_detector import BlinkDetector
from app.config import EAR_THRESHOLD
from tests.fakes import make_face, feed

OPEN = make_face(ear=0.30)
CLOSED = make_face(ear=0.10)

# A driver whose eyes are open wider than the fixed threshold, and one
# whose resting aperture sits below it.
NARROW_OPEN = make_face(ear=0.20)
NARROW_CLOSED = make_face(ear=0.06)


def blink_for(detector, open_face, closed_face, seconds, start=0.0):
    """Normal blinking: open for 0.8 s, closed for 0.2 s, repeated."""

    t = start

    for _ in range(int(seconds)):
        _, t = feed(detector, open_face, t, 0.8)
        _, t = feed(detector, closed_face, t, 0.2)

    return t


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


class AdaptiveThresholdTest(unittest.TestCase):
    """
    A fixed EAR threshold assumes an eye shape. These cover the driver
    it assumes wrong.
    """

    def test_the_fixed_threshold_misreads_a_narrow_eyed_driver(self):
        """
        With their eyes wide open, this driver reads as closed: PERCLOS
        pegs at 1.0 and the microsleep alert never stops.
        """

        detector = BlinkDetector(adaptive=False)
        result, _ = feed(detector, NARROW_OPEN, 0.0, 1.0)

        self.assertTrue(result["eyes_closed"])
        self.assertEqual(result["threshold"], EAR_THRESHOLD)

    def test_calibration_fixes_it(self):
        detector = BlinkDetector(adaptive=True)
        t = blink_for(detector, NARROW_OPEN, NARROW_CLOSED, 12)

        result, _ = feed(detector, NARROW_OPEN, t, 0.2)

        self.assertTrue(result["calibrated"])
        self.assertFalse(result["eyes_closed"])
        self.assertLess(result["threshold"], 0.20)

    def test_a_calibrated_narrow_eyed_driver_still_blinks(self):
        detector = BlinkDetector(adaptive=True)
        t = blink_for(detector, NARROW_OPEN, NARROW_CLOSED, 12)

        before = detector.blink_count

        _, t = feed(detector, NARROW_CLOSED, t, 0.2)
        result, _ = feed(detector, NARROW_OPEN, t, 0.5)

        self.assertEqual(result["blink_count"], before + 1)

    def test_two_drivers_get_two_thresholds(self):
        narrow = BlinkDetector(adaptive=True)
        blink_for(narrow, NARROW_OPEN, NARROW_CLOSED, 12)

        wide = BlinkDetector(adaptive=True)
        blink_for(wide, OPEN, CLOSED, 12)

        self.assertLess(narrow.calibrator.threshold,
                        wide.calibrator.threshold)

    def test_the_default_threshold_applies_while_calibrating(self):
        detector = BlinkDetector(adaptive=True)
        result, _ = feed(detector, OPEN, 0.0, 1.0)

        self.assertFalse(result["calibrated"])
        self.assertEqual(result["threshold"], EAR_THRESHOLD)

    def test_recalibrating_starts_again(self):
        detector = BlinkDetector(adaptive=True)
        t = blink_for(detector, NARROW_OPEN, NARROW_CLOSED, 12)

        detector.recalibrate()
        result, _ = feed(detector, NARROW_OPEN, t, 0.1)

        self.assertFalse(result["calibrated"])
        self.assertEqual(result["threshold"], EAR_THRESHOLD)

    def test_the_fixed_threshold_is_still_available(self):
        detector = BlinkDetector(adaptive=False)
        blink_for(detector, OPEN, CLOSED, 12)

        result, _ = feed(detector, OPEN, 20.0, 0.1)

        self.assertEqual(result["threshold"], EAR_THRESHOLD)
        self.assertEqual(result["eye_confidence"], 1.0)


if __name__ == "__main__":
    unittest.main()

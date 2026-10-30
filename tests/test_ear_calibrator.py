import unittest

from analytics.ear_calibrator import EarCalibrator
from app.config import EAR_THRESHOLD, EAR_THRESHOLD_MIN, EAR_THRESHOLD_MAX


def run(calibrator, pattern, duration=12.0, fps=30.0):
    """
    Feed `pattern(i)` as the EAR of frame i for `duration` seconds.
    Returns the last threshold the calibrator handed back.
    """

    threshold = None
    steps = int(round(duration * fps))

    for i in range(steps):
        threshold = calibrator.add(pattern(i), i / fps)

    return threshold


def blinking(open_ear, closed_ear, period=90, closed_frames=6):
    """A driver who blinks every `period` frames."""

    return lambda i: closed_ear if i % period < closed_frames else open_ear


class EarCalibratorTest(unittest.TestCase):

    def test_default_threshold_until_calibrated(self):
        calibrator = EarCalibrator()

        self.assertEqual(calibrator.add(0.30, 0.0), EAR_THRESHOLD)
        self.assertFalse(calibrator.calibrated)

    def test_two_ended_threshold_sits_inside_the_drivers_range(self):
        calibrator = EarCalibrator()
        run(calibrator, blinking(0.36, 0.09))

        self.assertTrue(calibrator.calibrated)
        self.assertEqual(calibrator.method, "two-ended")
        self.assertLess(calibrator.threshold, calibrator.open_ear)
        self.assertGreater(calibrator.threshold, calibrator.closed_ear)
        self.assertEqual(calibrator.confidence,
                         EarCalibrator.CONFIDENCE_TWO_ENDED)

    def test_narrow_eyes_get_a_lower_threshold_than_wide_ones(self):
        """
        The whole point of calibrating: the threshold has to follow the
        driver's own eye aperture, not a constant.
        """

        narrow = EarCalibrator()
        run(narrow, blinking(0.20, 0.06))

        wide = EarCalibrator()
        run(wide, blinking(0.42, 0.12))

        self.assertLess(narrow.threshold, wide.threshold)

    def test_narrow_eyed_driver_is_not_read_as_permanently_closed(self):
        """
        An open-eye EAR below the fixed threshold is the failure this
        exists to prevent: the driver reads as closed from the moment
        they sit down, PERCLOS pegs at 1.0 and the microsleep alert
        never stops.
        """

        open_ear = 0.20

        self.assertLess(open_ear, EAR_THRESHOLD)

        calibrator = EarCalibrator()
        run(calibrator, blinking(open_ear, 0.06))

        self.assertLess(calibrator.threshold, open_ear)

    def test_no_blink_in_the_window_falls_back_to_a_fraction_of_open(self):
        calibrator = EarCalibrator()
        run(calibrator, lambda i: 0.30)

        self.assertEqual(calibrator.method, "open-only")
        self.assertIsNone(calibrator.closed_ear)
        self.assertAlmostEqual(calibrator.threshold, 0.75 * 0.30, places=2)
        self.assertEqual(calibrator.confidence,
                         EarCalibrator.CONFIDENCE_OPEN_ONLY)

    def test_threshold_is_clamped_and_lowers_confidence(self):
        calibrator = EarCalibrator()
        run(calibrator, lambda i: 0.90)

        self.assertEqual(calibrator.threshold, EAR_THRESHOLD_MAX)
        self.assertLess(calibrator.confidence,
                        EarCalibrator.CONFIDENCE_TWO_ENDED)

    def test_threshold_never_leaves_the_sane_range(self):
        for open_ear, closed_ear in [(0.05, 0.01), (0.9, 0.8), (0.5, 0.02)]:
            calibrator = EarCalibrator()
            run(calibrator, blinking(open_ear, closed_ear))

            self.assertGreaterEqual(calibrator.threshold, EAR_THRESHOLD_MIN)
            self.assertLessEqual(calibrator.threshold, EAR_THRESHOLD_MAX)

    def test_too_few_samples_keeps_the_default(self):
        calibrator = EarCalibrator()

        # Two frames spanning the whole window.
        calibrator.add(0.30, 0.0)
        calibrator.add(0.30, 99.0)

        self.assertTrue(calibrator.calibrated)
        self.assertEqual(calibrator.threshold, EAR_THRESHOLD)
        self.assertEqual(calibrator.confidence,
                         EarCalibrator.CONFIDENCE_DEFAULT)

    def test_a_blink_does_not_drag_the_open_baseline_down(self):
        calibrator = EarCalibrator()
        run(calibrator, blinking(0.36, 0.09))

        self.assertAlmostEqual(calibrator.open_ear, 0.36, places=2)

    def test_restarting_discards_the_baseline(self):
        calibrator = EarCalibrator()
        run(calibrator, blinking(0.36, 0.09))

        calibrator.start()

        self.assertFalse(calibrator.calibrated)
        self.assertEqual(calibrator.threshold, EAR_THRESHOLD)
        self.assertIsNone(calibrator.open_ear)

    def test_the_baseline_is_frozen_once_learned(self):
        """
        A baseline that kept adapting would follow a driver whose eyes
        slowly narrow with fatigue, normalizing away the drowsiness it
        is meant to detect.
        """

        calibrator = EarCalibrator()
        run(calibrator, blinking(0.36, 0.09))

        learned = calibrator.threshold

        for i in range(600):
            calibrator.add(0.12, 20.0 + i / 30.0)

        self.assertEqual(calibrator.threshold, learned)


if __name__ == "__main__":
    unittest.main()

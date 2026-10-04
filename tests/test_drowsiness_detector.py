import unittest

from analytics.drowsiness_detector import DrowsinessDetector


def run_pattern(detector, start, seconds, closed_fraction, cycle=2.0,
                fps=30.0):
    """
    Feed eye states where, in each `cycle` seconds, the eyes are closed
    for the last `closed_fraction` of the cycle. Returns (result, t).
    """

    result = None
    t = start
    closed_since = None
    frames = int(round(seconds * fps))

    for i in range(frames):
        phase = (i / fps) % cycle
        closed = phase >= cycle * (1 - closed_fraction)

        if closed and closed_since is None:
            closed_since = t
        elif not closed:
            closed_since = None

        blink = {
            "eyes_closed": closed,
            "closed_duration": t - closed_since if closed else 0.0
        }

        result = detector.update(blink, t)
        t += 1 / fps

    return result, t


class DrowsinessDetectorTest(unittest.TestCase):

    def setUp(self):
        self.detector = DrowsinessDetector()

    def test_alert_driver(self):
        result, _ = run_pattern(self.detector, 0.0, 20.0, 0.05)
        self.assertTrue(result["perclos_ready"])
        self.assertAlmostEqual(result["perclos"], 0.05, delta=0.02)
        self.assertEqual(result["state"], "AWAKE")

    def test_tired_driver(self):
        result, _ = run_pattern(self.detector, 0.0, 30.0, 0.20)
        self.assertAlmostEqual(result["perclos"], 0.20, delta=0.02)
        self.assertEqual(result["state"], "TIRED")

    def test_drowsy_driver(self):
        result, _ = run_pattern(self.detector, 0.0, 30.0, 0.35)
        self.assertAlmostEqual(result["perclos"], 0.35, delta=0.02)
        self.assertEqual(result["state"], "DROWSY")

    def test_not_judged_before_enough_data(self):
        result, _ = run_pattern(self.detector, 0.0, 5.0, 0.40)
        self.assertFalse(result["perclos_ready"])
        self.assertEqual(result["state"], "AWAKE")

    def test_microsleep(self):
        _, t = run_pattern(self.detector, 0.0, 3.0, 0.0)

        result = self.detector.update(
            {"eyes_closed": True, "closed_duration": 1.6}, t
        )
        self.assertTrue(result["microsleep"])
        self.assertEqual(result["state"], "MICROSLEEP")
        self.assertEqual(result["microsleep_count"], 1)

        # Still the same episode.
        result = self.detector.update(
            {"eyes_closed": True, "closed_duration": 1.7}, t + 0.1
        )
        self.assertEqual(result["microsleep_count"], 1)

        result = self.detector.update(
            {"eyes_closed": False, "closed_duration": 0.0}, t + 0.2
        )
        self.assertFalse(result["microsleep"])
        self.assertEqual(result["microsleep_count"], 1)

    def test_old_closures_leave_the_window(self):
        _, t = run_pattern(self.detector, 0.0, 30.0, 0.40)
        result, _ = run_pattern(self.detector, t, 70.0, 0.0)
        self.assertAlmostEqual(result["perclos"], 0.0, places=6)
        self.assertEqual(result["state"], "AWAKE")

    def test_time_without_face_is_not_counted(self):
        _, t = run_pattern(self.detector, 0.0, 12.0, 0.0)
        self.assertIsNone(self.detector.update(None, t))

        before = self.detector.total_time
        self.detector.update(
            {"eyes_closed": True, "closed_duration": 0.0}, t + 5.0
        )

        # The 5 s gap is not added as closed (or open) time.
        self.assertAlmostEqual(self.detector.total_time, before, places=6)


if __name__ == "__main__":
    unittest.main()

import unittest

from analytics.distraction_detector import DistractionDetector


def pose(yaw=0.0, pitch=0.0, calibrated=True):
    return {"yaw": yaw, "pitch": pitch, "calibrated": calibrated}


def hold(detector, head_pose, gaze, start, seconds, fps=30.0):

    result = None
    t = start

    for _ in range(int(round(seconds * fps))):
        result = detector.update(head_pose, gaze, t)
        t += 1 / fps

    return result, t


class DistractionDetectorTest(unittest.TestCase):

    def setUp(self):
        self.detector = DistractionDetector()

    def test_looking_ahead(self):
        result, _ = hold(self.detector, pose(), None, 0.0, 3.0)
        self.assertTrue(result["eyes_on_road"])
        self.assertFalse(result["distracted"])

    def test_short_glance_is_not_a_distraction(self):
        result, _ = hold(self.detector, pose(yaw=40), None, 0.0, 1.0)
        self.assertFalse(result["eyes_on_road"])
        self.assertFalse(result["distracted"])
        self.assertEqual(result["distraction_count"], 0)

    def test_sustained_look_away_is_a_distraction(self):
        result, t = hold(self.detector, pose(yaw=-40), None, 0.0, 2.5)
        self.assertTrue(result["distracted"])
        self.assertEqual(result["distraction_count"], 1)

        result, _ = hold(self.detector, pose(), None, t, 1.0)
        self.assertTrue(result["eyes_on_road"])
        self.assertFalse(result["distracted"])
        self.assertEqual(result["distraction_count"], 1)

    def test_looking_down_is_off_road(self):
        result, _ = hold(self.detector, pose(pitch=30), None, 0.0, 2.5)
        self.assertTrue(result["distracted"])

    def test_eye_gaze_overrides_head_yaw(self):
        # Head turned, eyes back on the road.
        result, _ = hold(self.detector, pose(yaw=35), {"gaze_yaw": 5.0},
                         0.0, 3.0)
        self.assertTrue(result["eyes_on_road"])
        self.assertAlmostEqual(result["attention_yaw"], 5.0)

    def test_not_judged_while_calibrating(self):
        result = self.detector.update(pose(yaw=40, calibrated=False),
                                      None, 0.0)
        self.assertIsNone(result)

class FaceGapTest(unittest.TestCase):
    """
    What a missing face means while the driver is already looking away.

    It is not evidence they looked back. A head turned far enough to
    be a distraction is a head the landmarker often cannot find, so the
    gaps land in the middle of the very episodes this detector exists
    to catch. Measured on a recording: a nine second turn lost two and
    a quarter seconds to a single two-thirds-of-a-second gap, because
    the clock started again afterwards and never ran long enough.

    So the clock is held across a short gap, and nothing is claimed
    while it lasts.
    """

    def setUp(self):
        self.detector = DistractionDetector()

    def blind(self, start, seconds, fps=30.0):
        t = start

        for _ in range(int(round(seconds * fps))):
            self.assertIsNone(self.detector.update(None, None, t))
            t += 1 / fps

        return t

    def test_nothing_is_reported_while_the_face_is_missing(self):
        """Holding the clock is not the same as claiming to see."""

        _, t = hold(self.detector, pose(yaw=40), None, 0.0, 2.5)
        self.assertTrue(self.detector.distracted_active)

        self.blind(t, 0.3)

        self.assertFalse(self.detector.distracted_active)

    def test_a_brief_gap_keeps_the_clock_running(self):
        _, t = hold(self.detector, pose(yaw=40), None, 0.0, 1.5)
        t = self.blind(t, 0.5)
        result, _ = hold(self.detector, pose(yaw=40), None, t, 0.5)

        self.assertTrue(result["distracted"])
        self.assertEqual(result["distraction_count"], 1)

    def test_a_long_gap_abandons_it(self):
        """
        Past the grace the face has been gone too long to assume
        anything about where it was pointing.
        """

        _, t = hold(self.detector, pose(yaw=40), None, 0.0, 1.5)
        t = self.blind(t, 1.5)
        result, _ = hold(self.detector, pose(yaw=40), None, t, 1.0)

        self.assertFalse(result["distracted"])

    def test_the_gap_counts_toward_the_dwell(self):
        """
        The driver was off the road either side of it, so the time
        between is off-road time too.
        """

        _, t = hold(self.detector, pose(yaw=40), None, 0.0, 1.0)
        t = self.blind(t, 0.8)
        result, _ = hold(self.detector, pose(yaw=40), None, t, 0.4)

        self.assertTrue(result["distracted"])

    def test_looking_back_after_a_gap_still_clears_it(self):
        """
        The hold is about absent evidence, not about overriding it.
        """

        _, t = hold(self.detector, pose(yaw=40), None, 0.0, 1.9)
        t = self.blind(t, 0.3)
        _, t = hold(self.detector, pose(yaw=0), None, t, 0.3)
        result, _ = hold(self.detector, pose(yaw=40), None, t, 1.0)

        self.assertFalse(result["distracted"])

    def test_a_gap_with_no_face_ever_seen_does_not_crash(self):
        """A clip that opens on an empty seat."""

        self.blind(0.0, 2.0)

        result, _ = hold(self.detector, pose(yaw=40), None, 2.0, 2.5)

        self.assertTrue(result["distracted"])


if __name__ == "__main__":
    unittest.main()

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

    def test_face_lost_resets_timer(self):
        _, t = hold(self.detector, pose(yaw=40), None, 0.0, 1.5)
        self.assertIsNone(self.detector.update(None, None, t))
        result, _ = hold(self.detector, pose(yaw=40), None, t + 0.1, 1.0)
        self.assertFalse(result["distracted"])


if __name__ == "__main__":
    unittest.main()

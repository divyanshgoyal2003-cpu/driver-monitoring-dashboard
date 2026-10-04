import math
import unittest

from analytics.gaze_estimator import GazeEstimator
from tests.fakes import WIDTH, HEIGHT, blank_face, set_irises


def gaze_for(ratio, head_yaw=None):

    face = set_irises(blank_face(), ratio)
    head_pose = {"yaw": head_yaw} if head_yaw is not None else None

    return GazeEstimator().estimate([face], WIDTH, HEIGHT, head_pose)


class GazeEstimatorTest(unittest.TestCase):

    def test_centred_iris_looks_ahead(self):
        result = gaze_for(0.5)
        self.assertAlmostEqual(result["iris_ratio"], 0.5, places=6)
        self.assertAlmostEqual(result["eye_yaw"], 0.0, places=6)
        self.assertEqual(result["direction"], "CENTER")

    def test_iris_toward_image_right_is_driver_left(self):
        result = gaze_for(0.7)
        # asin(0.2 / 0.4) = 30 degrees
        self.assertAlmostEqual(result["eye_yaw"], -30.0, places=3)
        self.assertEqual(result["direction"], "LEFT")

    def test_iris_toward_image_left_is_driver_right(self):
        result = gaze_for(0.3)
        self.assertAlmostEqual(result["eye_yaw"], 30.0, places=3)
        self.assertEqual(result["direction"], "RIGHT")

    def test_eyes_compensating_for_head_turn_look_ahead(self):
        # Head turned 17.5 degrees left, eyes turned back 17.5 right.
        ratio = 0.5 - 0.4 * math.sin(math.radians(17.5))
        result = gaze_for(ratio, head_yaw=-17.5)
        self.assertAlmostEqual(result["gaze_yaw"], 0.0, places=3)
        self.assertEqual(result["direction"], "CENTER")

    def test_head_and_eyes_turned_together_add_up(self):
        result = gaze_for(0.6, head_yaw=-10)
        # asin(0.1 / 0.4) = 14.48 degrees
        self.assertAlmostEqual(result["gaze_yaw"], -24.48, places=1)
        self.assertEqual(result["direction"], "LEFT")

    def test_extreme_ratio_is_clamped(self):
        result = gaze_for(1.2)
        self.assertAlmostEqual(result["eye_yaw"], -90.0, places=3)

    def test_model_without_iris_landmarks(self):
        face = blank_face()[:468]
        self.assertIsNone(GazeEstimator().estimate([face], WIDTH, HEIGHT))

    def test_no_face(self):
        self.assertIsNone(GazeEstimator().estimate([], WIDTH, HEIGHT))


if __name__ == "__main__":
    unittest.main()

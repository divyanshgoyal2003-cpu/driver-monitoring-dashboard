import itertools
import unittest

from analytics.head_pose_estimator import HeadPoseEstimator
from tests.fakes import WIDTH, HEIGHT, posed_face


class HeadPoseEstimatorTest(unittest.TestCase):

    def setUp(self):
        self.estimator = HeadPoseEstimator(WIDTH, HEIGHT)

    def estimate(self, **angles):
        return self.estimator.estimate([posed_face(**angles)])

    def test_frontal_face_is_forward(self):
        # Regression: a frontal face used to decompose to yaw = 180.
        result = self.estimate()
        self.assertAlmostEqual(result["yaw"], 0.0, delta=0.5)
        self.assertAlmostEqual(result["pitch"], 0.0, delta=0.5)
        self.assertAlmostEqual(result["roll"], 0.0, delta=0.5)
        self.assertEqual(result["direction"], "FORWARD")

    def test_no_flips_near_frontal(self):
        for yaw, pitch, roll in itertools.product([-4, -1, 0, 1, 4],
                                                  repeat=3):
            result = self.estimate(yaw=yaw, pitch=pitch, roll=roll)
            self.assertAlmostEqual(result["yaw"], yaw, delta=1.0)
            self.assertAlmostEqual(result["pitch"], pitch, delta=1.0)
            self.assertAlmostEqual(result["roll"], roll, delta=1.0)
            self.assertEqual(result["direction"], "FORWARD")

    def test_directions(self):
        cases = [
            ({"yaw": -25}, "LEFT"),
            ({"yaw": 25}, "RIGHT"),
            ({"pitch": -20}, "UP"),
            ({"pitch": 20}, "DOWN"),
        ]

        for angles, expected in cases:
            result = self.estimate(**angles)
            self.assertEqual(result["direction"], expected, angles)

            for name, value in angles.items():
                self.assertAlmostEqual(result[name], value, delta=1.0)

    def test_combined_rotation(self):
        result = self.estimate(yaw=30, pitch=-12, roll=8)
        self.assertAlmostEqual(result["yaw"], 30, delta=1.0)
        self.assertAlmostEqual(result["pitch"], -12, delta=1.0)
        self.assertAlmostEqual(result["roll"], 8, delta=1.0)

    def test_no_calibration_without_timestamp(self):
        result = self.estimate(yaw=10)
        self.assertFalse(result["calibrated"])
        self.assertAlmostEqual(result["yaw"], 10, delta=1.0)

    def test_calibration_removes_camera_offset(self):
        # Camera mounted off-centre: the driver looking at the road
        # reads as yaw 10, pitch 8.
        neutral = posed_face(yaw=10, pitch=8)

        t = 0.0
        result = None

        while t < 3.5:
            result = self.estimator.estimate([neutral], t)
            t += 1 / 30

        self.assertTrue(result["calibrated"])
        self.assertAlmostEqual(result["yaw"], 0.0, delta=1.0)
        self.assertAlmostEqual(result["pitch"], 0.0, delta=1.0)
        self.assertEqual(result["direction"], "FORWARD")

        turned = self.estimator.estimate([posed_face(yaw=35, pitch=8)], t)
        self.assertAlmostEqual(turned["yaw"], 25, delta=1.0)
        self.assertEqual(turned["direction"], "RIGHT")

    def test_calibration_offset_is_capped(self):
        face = posed_face(yaw=40)

        t = 0.0

        while t < 3.5:
            result = self.estimator.estimate([face], t)
            t += 1 / 30

        self.assertAlmostEqual(self.estimator.yaw_offset, 25.0, delta=0.01)
        self.assertAlmostEqual(result["yaw"], 15.0, delta=1.0)

    def test_recalibration(self):
        t = 0.0

        while t < 3.5:
            self.estimator.estimate([posed_face(yaw=10)], t)
            t += 1 / 30

        self.estimator.start_calibration()
        result = self.estimator.estimate([posed_face(yaw=-5)], t)
        self.assertFalse(result["calibrated"])

        start = t

        while t < start + 3.5:
            result = self.estimator.estimate([posed_face(yaw=-5)], t)
            t += 1 / 30

        self.assertTrue(result["calibrated"])
        self.assertAlmostEqual(self.estimator.yaw_offset, -5.0, delta=1.0)


if __name__ == "__main__":
    unittest.main()

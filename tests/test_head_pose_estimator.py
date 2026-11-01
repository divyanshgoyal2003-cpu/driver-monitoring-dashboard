import itertools
import unittest

from analytics.head_pose_estimator import HeadPoseEstimator
from tests.fakes import WIDTH, HEIGHT, posed_face, shaped_face


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


class MirrorPoseTest(unittest.TestCase):
    """
    A turned head must not come back turned the other way.

    Six facial landmarks are nearly coplanar, and a near-planar point
    set has a second pose -- its mirror about the image plane -- that
    reprojects almost as well as the real one. A local optimiser picks
    whichever its starting guess is nearer to, so the sign of a large
    yaw is decided by the shape of the driver's face rather than by
    which way they are looking.

    Measured on a real recording before this was fixed: a driver
    holding a steady 63 degree turn was reported crossing between +63
    and -75 thirty-one times in 206 frames. Distraction needs an
    unbroken dwell, so nine seconds of looking away scored as 2.5 and
    recall over the clip came out at 27%.

    These faces are deliberately not the estimator's model, because
    with the canonical head at an orthographic distance every solver
    gets this right and the bug is invisible.
    """

    # Well past the distraction threshold, and far enough round for the
    # two solutions to be distinct answers rather than a rounding
    # difference.
    TURNS = (40, 50, 60, 70, -40, -50, -60, -70)

    FACES = 25

    def setUp(self):
        self.estimator = HeadPoseEstimator(WIDTH, HEIGHT)

    def test_a_turned_head_is_never_reported_the_other_way(self):
        for seed in range(self.FACES):
            for yaw in self.TURNS:
                face = shaped_face(yaw=yaw, seed=seed)
                measured = self.estimator.estimate([face])["yaw"]

                self.assertGreater(
                    measured * yaw, 0,
                    f"face {seed} turned {yaw} deg was read as {measured:.1f}"
                )

    def test_the_angle_survives_an_unfamiliar_face(self):
        """
        Fitting the wrong head shape costs accuracy, which is expected
        and is what calibration absorbs. It must not cost the answer.
        """

        for seed in range(self.FACES):
            for yaw in self.TURNS:
                face = shaped_face(yaw=yaw, seed=seed)
                measured = self.estimator.estimate([face])["yaw"]

                self.assertAlmostEqual(
                    measured, yaw, delta=15.0,
                    msg=f"face {seed} turned {yaw} deg read {measured:.1f}"
                )

    def test_a_turn_is_called_the_right_way_round(self):
        for seed in range(self.FACES):
            self.assertEqual(
                self.estimator.estimate([shaped_face(yaw=50, seed=seed)]
                                        )["direction"], "RIGHT", seed)
            self.assertEqual(
                self.estimator.estimate([shaped_face(yaw=-50, seed=seed)]
                                        )["direction"], "LEFT", seed)

    def test_a_head_turning_smoothly_reads_smoothly(self):
        """
        The failure that mattered was not one wrong frame, it was the
        discontinuity. A dwell timer is only reset by a reading that
        disagrees with its neighbours, so a solver that crosses to the
        mirror solution halfway through a turn defeats the timer even
        while most of its answers are right.

        Half a degree of real movement must not move the estimate by
        ten.
        """

        for seed in range(self.FACES):
            readings = [
                self.estimator.estimate(
                    [shaped_face(yaw=step / 2.0, seed=seed)]
                )["yaw"]
                for step in range(151)
            ]

            for before, after in zip(readings, readings[1:]):
                self.assertLess(
                    abs(after - before), 10.0,
                    f"face {seed} jumped {before:.1f} -> {after:.1f}"
                )

    def test_a_close_frontal_face_is_still_forward(self):
        """The fix must not buy large angles at the cost of small ones."""

        for seed in range(self.FACES):
            result = self.estimator.estimate([shaped_face(yaw=0, seed=seed)])

            self.assertAlmostEqual(result["yaw"], 0.0, delta=2.0)
            self.assertEqual(result["direction"], "FORWARD")


if __name__ == "__main__":
    unittest.main()

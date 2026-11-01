import itertools
import unittest

from analytics.head_pose_estimator import HeadPoseEstimator
from tests.fakes import (
    WIDTH,
    HEIGHT,
    RECORDED_DISTANCE,
    open_mouth_face,
    posed_face,
    shaped_face,
)


def calibrated_on(estimator, face, seconds=4.0, fps=30.0):
    """
    Let the estimator learn its neutral pose from `face`, the way it
    does from the first seconds of a drive, and return the reading it
    settles on.
    """

    result = None
    t = 0.0

    for _ in range(int(seconds * fps)):
        result = estimator.estimate([face], t)
        t += 1 / fps

    return result


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
        """
        The fix must not buy large angles at the cost of small ones.

        Yaw is checked raw, because nothing should bend it sideways.
        Direction is checked after calibration, which is the only form
        the system ever promises it in: fitting a generic head to a
        particular one leaves a standing pitch offset of a few
        degrees, and learning the driver's neutral pose is exactly how
        that is removed.
        """

        for seed in range(self.FACES):
            face = shaped_face(yaw=0, seed=seed)

            self.assertAlmostEqual(
                self.estimator.estimate([face])["yaw"], 0.0, delta=2.0)

            estimator = HeadPoseEstimator(WIDTH, HEIGHT)
            result = calibrated_on(estimator, face)

            self.assertEqual(result["direction"], "FORWARD", seed)

    def test_the_standing_offset_is_inside_what_calibration_can_remove(self):
        """
        Calibration clamps what it will believe, so a bias beyond that
        would survive it and tilt every later reading.
        """

        for seed in range(self.FACES):
            estimator = HeadPoseEstimator(WIDTH, HEIGHT)
            raw = estimator.estimate([shaped_face(yaw=0, seed=seed)])

            self.assertLess(abs(raw["pitch"]), estimator.max_offset, seed)
            self.assertLess(abs(raw["yaw"]), estimator.max_offset, seed)


class OpenMouthTest(unittest.TestCase):
    """
    What a yawn does to a solver that assumes a rigid head.

    It assumes one because solvePnP has to: it is handed a fixed 3D
    model and asked which rotation of it best explains the landmarks.
    A mouth opening breaks that assumption outright. Measured on a
    recording in which the video shows the driver facing straight
    ahead throughout, the reported yaw reached -28 degrees off its own
    baseline and held there for over two seconds -- long enough to be
    logged as a distraction that never happened.

    Dropping the chin from the solve helps, because it is the single
    largest displacement, but it is not a cure. Most of the residual
    is the eye corners, which move asymmetrically too: holding the
    left one still takes the measured yaw from -24.9 back to 0.8, and
    no choice of model points can hold a landmark still. So the
    estimator reports that it should not be believed, and the
    detectors downstream treat that like a face they cannot see.
    """

    FACES = 15

    def test_the_chin_is_not_solved_against(self):
        """
        It travels 121 pixels in a measured yawn, four times further
        than anything else, with the longest lever arm in the model.
        """

        estimator = HeadPoseEstimator(WIDTH, HEIGHT)

        self.assertNotIn(HeadPoseEstimator.CHIN,
                         HeadPoseEstimator.LANDMARKS)

        self.assertEqual(len(HeadPoseEstimator.LANDMARKS),
                         len(estimator.model_points))

    def test_a_closed_mouth_is_believed(self):
        for seed in range(self.FACES):
            estimator = HeadPoseEstimator(WIDTH, HEIGHT)
            result = estimator.estimate(
                [open_mouth_face(yaw=0, openness=0.0, seed=seed)])

            self.assertTrue(result["reliable"], seed)

    def test_a_yawn_is_not_believed(self):
        for seed in range(self.FACES):
            estimator = HeadPoseEstimator(WIDTH, HEIGHT)
            result = estimator.estimate(
                [open_mouth_face(yaw=0, openness=1.0, seed=seed)])

            self.assertFalse(result["reliable"], seed)

    def test_talking_is_still_believed(self):
        """
        A mouth moving in speech is nothing like one at full stretch,
        and blinding the estimator through a conversation would cost
        far more than it saved.
        """

        for seed in range(self.FACES):
            estimator = HeadPoseEstimator(WIDTH, HEIGHT)
            result = estimator.estimate(
                [open_mouth_face(yaw=0, openness=0.25, seed=seed)])

            self.assertTrue(result["reliable"], seed)

    def test_a_turned_head_is_still_read_through_a_closed_mouth(self):
        """The gate must key on the mouth, not on the angle."""

        for seed in range(self.FACES):
            estimator = HeadPoseEstimator(WIDTH, HEIGHT)
            result = estimator.estimate(
                [shaped_face(yaw=55, seed=seed, distance=RECORDED_DISTANCE)])

            self.assertTrue(result["reliable"], seed)
            self.assertGreater(result["yaw"], 0, seed)

    def test_a_yawn_does_not_teach_a_crooked_neutral_pose(self):
        """
        Calibration takes the median of its samples, so a driver who
        yawns through it would have the deformation baked into every
        later reading.
        """

        estimator = HeadPoseEstimator(WIDTH, HEIGHT)
        t = 0.0

        for _ in range(120):
            estimator.estimate([open_mouth_face(yaw=0, seed=2)], t)
            t += 1 / 30

        self.assertFalse(estimator.calibrated)


if __name__ == "__main__":
    unittest.main()

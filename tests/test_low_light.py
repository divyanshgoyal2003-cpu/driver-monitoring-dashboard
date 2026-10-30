"""
Low-light enhancement.

The point of these is the two promises the rest of the system relies
on: enhancement is a ramp rather than a switch, and it never reports
more light than the camera actually saw.
"""

import unittest

import numpy as np

from utils.low_light import LowLightEnhancer


def enhancer():
    return LowLightEnhancer(threshold=70.0, clip_limit=3.0, grid=8)


def frame(level, noise=12, size=256):
    """
    A frame around `level` with some texture to redistribute.

    Kept large enough that a CLAHE tile holds a useful number of
    pixels: on a tiny frame every clip limit rounds to the same floor
    and the strength ramp stops being visible in the output.
    """

    rng = np.random.default_rng(7)

    pixels = rng.normal(level, noise, (size, size, 3))

    return np.clip(pixels, 0, 255).astype(np.uint8)


class StrengthTest(unittest.TestCase):

    def test_a_lit_frame_is_left_alone(self):
        self.assertEqual(enhancer().strength(120.0), 0.0)

    def test_the_threshold_itself_is_left_alone(self):
        self.assertEqual(enhancer().strength(70.0), 0.0)

    def test_a_very_dim_frame_gets_full_strength(self):
        self.assertEqual(enhancer().strength(10.0), 1.0)

    def test_strength_rises_as_the_frame_dims(self):
        light = enhancer().strength(65.0)
        dim = enhancer().strength(40.0)

        self.assertLess(light, dim)
        self.assertLessEqual(dim, 1.0)

    def test_strength_is_continuous_around_the_threshold(self):
        """
        A step here would make a frame hovering at the boundary flip
        between enhanced and raw, and the detectors flicker with it.
        """

        just_under = enhancer().strength(69.9)

        self.assertGreater(just_under, 0.0)
        self.assertLess(just_under, 0.05)

    def test_a_missing_reading_is_not_enhanced(self):
        self.assertEqual(enhancer().strength(None), 0.0)


class EnhanceTest(unittest.TestCase):

    def test_a_lit_frame_is_returned_untouched(self):
        original = frame(140)

        self.assertIs(enhancer().enhance(original, 140.0), original)

    def test_a_dim_frame_gains_contrast(self):
        original = frame(30)
        lifted = enhancer().enhance(original, 30.0)

        self.assertGreater(lifted.std(), original.std())

    def test_the_original_frame_is_not_modified(self):
        original = frame(30)
        before = original.copy()

        enhancer().enhance(original, 30.0)

        self.assertTrue(np.array_equal(original, before))

    def test_the_shape_and_type_survive(self):
        original = frame(30)
        lifted = enhancer().enhance(original, 30.0)

        self.assertEqual(lifted.shape, original.shape)
        self.assertEqual(lifted.dtype, original.dtype)

    def test_colour_survives(self):
        """
        LAB rather than grayscale: the emotion and phone models want a
        colour frame.
        """

        original = frame(30)
        lifted = enhancer().enhance(original, 30.0)

        self.assertEqual(lifted.ndim, 3)
        self.assertEqual(lifted.shape[2], 3)

    def test_a_dimmer_frame_is_lifted_harder(self):
        dim = enhancer().enhance(frame(50), 50.0)
        darker = enhancer().enhance(frame(50), 20.0)

        # Same pixels, different claimed brightness: only the strength
        # of the lift differs.
        self.assertGreater(darker.std(), dim.std())

    def test_the_clahe_object_is_reused_between_similar_frames(self):
        lifter = enhancer()

        lifter.enhance(frame(30), 30.0)
        first = lifter._clahe

        lifter.enhance(frame(30), 30.0)

        self.assertIs(lifter._clahe, first)

    def test_the_clahe_object_is_rebuilt_when_the_light_changes(self):
        lifter = enhancer()

        lifter.enhance(frame(30), 15.0)
        first = lifter._clahe

        lifter.enhance(frame(30), 65.0)

        self.assertIsNot(lifter._clahe, first)


if __name__ == "__main__":
    unittest.main()

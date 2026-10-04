import unittest

import cv2
import numpy as np

from models.detectors.seatbelt_detector import SeatbeltDetector
from tests.test_pipeline import sample_image, sample_driver_frame

# YOLO face boxes in the sample frames (see tests/test_pipeline.py).
DRIVER_FACE = (619, 176, 876, 500)        # sample_driver_frame()
CALM_FACE = (913, 96, 1057, 278)          # sample_image(), right
SHOUTING_FACE = (541, 244, 664, 452)      # sample_image(), left


def draw_strap(image, face_box, direction=1, width_ratio=0.35,
               color=(95, 95, 95)):
    """
    Paint a seatbelt-like strap over the collarbone beside the neck and
    down to the opposite hip. direction=1 runs from the image-right
    shoulder down to the left; -1 is the mirror image.
    """

    x1, y1, x2, y2 = face_box
    fw, fh = x2 - x1, y2 - y1

    if direction > 0:
        start = np.array([x2 + 0.15 * fw, y2 + 0.15 * fh])
        end = np.array([x1 - 0.7 * fw, y2 + 2.6 * fh])
    else:
        start = np.array([x1 - 0.15 * fw, y2 + 0.15 * fh])
        end = np.array([x2 + 0.7 * fw, y2 + 2.6 * fh])

    along = end - start
    normal = np.array([-along[1], along[0]]) / np.linalg.norm(along)
    half = normal * width_ratio * fw / 2

    polygon = np.array(
        [start + half, end + half, end - half, start - half], np.int32
    )

    out = image.copy()
    cv2.fillPoly(out, [polygon], color)

    # Fabric texture, so the strap is not a perfectly flat fill.
    mask = np.zeros(out.shape[:2], np.uint8)
    cv2.fillPoly(mask, [polygon], 255)
    noise = np.random.default_rng(0).normal(0, 6, out.shape)
    textured = np.clip(out.astype(np.float64) + noise, 0, 255)
    out[mask > 0] = textured[mask > 0].astype(np.uint8)

    return out


def plain_scene(face_box, size=(720, 1280)):
    """Uniform grey scene with a darker face-sized blob."""

    image = np.full((*size, 3), 150, np.uint8)
    x1, y1, x2, y2 = face_box
    cv2.rectangle(image, (x1, y1), (x2, y2), (110, 120, 140), -1)

    return image


class SeatbeltDetectorTest(unittest.TestCase):

    def setUp(self):
        self.detector = SeatbeltDetector()
        self.driver = sample_driver_frame()
        self.photo = sample_image()

    def test_suit_without_belt_is_not_flagged(self):
        # Suit lapels, a shirt collar and a tie make long diagonal edges
        # too. (The calm face is not used here: he wears a lanyard, a
        # real diagonal strap that a shape-based detector cannot tell
        # from a seatbelt.)
        result = self.detector.detect(self.photo, SHOUTING_FACE)
        self.assertTrue(result["visible"])
        self.assertFalse(result["belt_found"])

    def test_close_up_without_chest_is_not_judged(self):
        # Only ~0.6 face heights of chest are in view: too little to see
        # a strap cross the chest, so no judgement either way.
        result = self.detector.detect(self.driver, DRIVER_FACE)
        self.assertFalse(result["visible"])
        self.assertFalse(result["belt_found"])

    def test_strap_is_found_in_both_orientations(self):
        face = (560, 150, 720, 350)

        cases = [
            (self.photo, CALM_FACE),
            (self.photo, SHOUTING_FACE),
            (plain_scene(face), face),
        ]

        for image, face in cases:
            for direction in (1, -1):
                belted = draw_strap(image, face, direction)
                result = self.detector.detect(belted, face)
                self.assertTrue(result["belt_found"], (face, direction))
                self.assertEqual(len(result["strap"]), 2)

    def test_strap_edges_are_reported_in_frame_coordinates(self):
        belted = draw_strap(self.photo, CALM_FACE)
        result = self.detector.detect(belted, CALM_FACE)

        rx1, ry1, rx2, ry2 = result["roi"]

        for x1, y1, x2, y2 in result["strap"]:
            for x, y in ((x1, y1), (x2, y2)):
                self.assertTrue(rx1 <= x <= rx2 and ry1 <= y <= ry2)

    def test_strap_width_must_be_plausible(self):
        face = (560, 150, 720, 350)
        scene = plain_scene(face)

        thin = draw_strap(scene, face, width_ratio=0.06)
        wide = draw_strap(scene, face, width_ratio=0.9)
        normal = draw_strap(scene, face, width_ratio=0.35)

        self.assertFalse(self.detector.detect(thin, face)["belt_found"])
        self.assertFalse(self.detector.detect(wide, face)["belt_found"])
        self.assertTrue(self.detector.detect(normal, face)["belt_found"])

    def test_lapels_are_ignored(self):
        face = (560, 150, 720, 350)
        scene = plain_scene(face)

        self._draw_lapels(scene)

        result = self.detector.detect(scene, face)
        self.assertTrue(result["visible"])
        self.assertFalse(result["belt_found"])

    def test_lapels_are_rejected_by_the_centre_line_rule(self):
        # Guard against the lapel test passing for another reason: with
        # the centre-line rule switched off, the lapels do look like a
        # strap.
        face = (560, 150, 720, 350)
        scene = plain_scene(face)
        self._draw_lapels(scene)

        self.detector._crosses = lambda a, b, midline: True

        self.assertTrue(self.detector.detect(scene, face)["belt_found"])

    @staticmethod
    def _draw_lapels(scene):
        """
        On each side of the centre line (x = 640), a band with two
        parallel edges a strap-width apart, running from beside the neck
        down to the centre, where the V closes. Same angle (~62 deg) and
        width as a strap, but it never crosses the centre.
        """

        for sign in (1, -1):
            for offset in (0, 55):
                top = (640 + sign * (150 + offset), 370)
                bottom = (640 + sign * (5 + offset), 640)
                cv2.line(scene, top, bottom, (40, 40, 40), 3)

    def test_plain_scene(self):
        face = (560, 150, 720, 350)
        result = self.detector.detect(plain_scene(face), face)
        self.assertTrue(result["visible"])
        self.assertFalse(result["belt_found"])

    def test_chest_out_of_view(self):
        # Face near the bottom of the frame: little chest below it.
        face = (560, 450, 720, 650)
        result = self.detector.detect(plain_scene(face), face)
        self.assertFalse(result["visible"])

    def test_no_face(self):
        result = self.detector.detect(self.driver, None)
        self.assertFalse(result["visible"])
        self.assertFalse(result["belt_found"])


if __name__ == "__main__":
    unittest.main()

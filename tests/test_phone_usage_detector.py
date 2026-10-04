import unittest

from analytics.phone_usage_detector import PhoneUsageDetector

PHONE = [{"label": "cell phone", "bbox": (10, 10, 50, 90),
          "confidence": 0.8}]


def run(detector, sequence, start=0.0, fps=30.0):
    """
    sequence: list of (seconds, phone visible). Returns (result, t).
    """

    result = None
    t = start

    for seconds, visible in sequence:
        for _ in range(int(round(seconds * fps))):
            result = detector.update(PHONE if visible else [], t)
            t += 1 / fps

    return result, t


class PhoneUsageDetectorTest(unittest.TestCase):

    def setUp(self):
        self.detector = PhoneUsageDetector()

    def test_no_phone(self):
        result, _ = run(self.detector, [(2.0, False)])
        self.assertFalse(result["detected"])
        self.assertFalse(result["in_use"])

    def test_brief_detection_is_not_usage(self):
        result, _ = run(self.detector, [(0.2, True)])
        self.assertTrue(result["detected"])
        self.assertFalse(result["in_use"])

        result, _ = run(self.detector, [(2.0, False)], start=0.2)
        self.assertFalse(result["in_use"])
        self.assertEqual(result["usage_count"], 0)

    def test_sustained_detection_is_usage(self):
        result, _ = run(self.detector, [(1.0, True)])
        self.assertTrue(result["in_use"])
        self.assertEqual(result["usage_count"], 1)
        self.assertGreaterEqual(result["usage_duration"], 0.9)

    def test_short_dropouts_do_not_end_usage(self):
        result, _ = run(self.detector, [
            (1.0, True), (0.5, False), (1.0, True), (0.5, False)
        ])
        self.assertTrue(result["in_use"])
        self.assertEqual(result["usage_count"], 1)
        self.assertGreaterEqual(result["usage_duration"], 2.9)

    def test_usage_ends_after_release_time(self):
        result, t = run(self.detector, [(1.0, True), (1.5, False)])
        self.assertFalse(result["in_use"])

        result, _ = run(self.detector, [(1.0, True)], start=t)
        self.assertTrue(result["in_use"])
        self.assertEqual(result["usage_count"], 2)


if __name__ == "__main__":
    unittest.main()

import unittest

from analytics.seatbelt_monitor import SeatbeltMonitor


def observation(visible=True, found=False):
    return {"visible": visible, "belt_found": found, "roi": None,
            "strap": None}


def run(monitor, seconds, start=0.0, fps=10.0, **kwargs):

    result = None
    t = start

    for _ in range(int(round(seconds * fps))):
        result = monitor.update(observation(**kwargs), t)
        t += 1 / fps

    return result, t


class SeatbeltMonitorTest(unittest.TestCase):

    def setUp(self):
        self.monitor = SeatbeltMonitor()

    def test_starts_unknown(self):
        result = self.monitor.update(observation(found=True), 0.0)
        self.assertEqual(result["status"], "UNKNOWN")

    def test_belt_on(self):
        result, _ = run(self.monitor, 2.0, found=True)
        self.assertEqual(result["status"], "ON")
        self.assertAlmostEqual(result["ratio"], 1.0)

    def test_belt_off(self):
        result, _ = run(self.monitor, 2.0, found=False)
        self.assertEqual(result["status"], "OFF")

    def test_chest_not_visible(self):
        result, _ = run(self.monitor, 2.0, visible=False)
        self.assertEqual(result["status"], "UNKNOWN")

    def test_borderline_ratio_keeps_previous_status(self):
        _, t = run(self.monitor, 4.0, found=True)

        # Alternate found / not found: ratio 0.5 -> stays ON...
        for i in range(30):
            result = self.monitor.update(observation(found=i % 3 == 0), t)
            t += 0.1

        # ...ratio ~0.33 is between OFF_RATIO and ON_RATIO, so ON holds.
        self.assertAlmostEqual(result["ratio"], 1 / 3, delta=0.05)
        self.assertEqual(result["status"], "ON")

    def test_switches_off_when_belt_disappears(self):
        _, t = run(self.monitor, 4.0, found=True)
        result, _ = run(self.monitor, 4.0, start=t, found=False)
        self.assertEqual(result["status"], "OFF")

    def test_frames_without_observation_keep_status(self):
        _, t = run(self.monitor, 2.0, found=True)
        result = self.monitor.update(None, t)
        self.assertEqual(result["status"], "ON")


if __name__ == "__main__":
    unittest.main()

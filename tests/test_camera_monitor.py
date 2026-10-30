"""
Telling a covered lens apart from an empty seat.

The monitor itself is tested first, then the thing it was built for:
that a blocked camera raises its own alert and stops the pipeline from
also reporting a missing driver.
"""

import unittest

from analytics.camera_monitor import CameraMonitor
from risk_engine.scoring import DriverSafetyIndex


def monitor():
    return CameraMonitor(
        max_brightness=15.0,
        max_variance=8.0,
        min_duration=2.0
    )


def feed(camera, brightness, variance, start, end, step=0.5):
    """Hold one reading from `start` to `end`, and return the last result."""

    result = None
    timestamp = start

    while timestamp <= end:
        result = camera.update(brightness, variance, timestamp)
        timestamp += step

    return result


class CameraMonitorTest(unittest.TestCase):

    def test_a_lit_cabin_is_not_blocked(self):
        result = feed(monitor(), 120.0, 45.0, 0.0, 10.0)

        self.assertFalse(result["blocked"])

    def test_a_covered_lens_is_blocked(self):
        result = feed(monitor(), 3.0, 1.0, 0.0, 10.0)

        self.assertTrue(result["blocked"])

    def test_a_dark_cabin_is_not_blocked(self):
        """
        The case that makes brightness alone useless: almost no light,
        but shapes and sensor noise still vary across the frame.
        """

        result = feed(monitor(), 8.0, 20.0, 0.0, 10.0)

        self.assertFalse(result["blocked"])

    def test_a_flat_bright_frame_is_not_blocked(self):
        """A wall or a white-out is not a cover."""

        result = feed(monitor(), 200.0, 2.0, 0.0, 10.0)

        self.assertFalse(result["blocked"])

    def test_a_brief_flat_patch_is_not_blocked(self):
        """A camera still waking up should not be accused."""

        camera = monitor()

        result = feed(camera, 2.0, 1.0, 0.0, 1.5)

        self.assertFalse(result["blocked"])

    def test_blocking_is_reported_once_the_duration_passes(self):
        camera = monitor()

        self.assertFalse(camera.update(2.0, 1.0, 0.0)["blocked"])
        self.assertFalse(camera.update(2.0, 1.0, 1.9)["blocked"])
        self.assertTrue(camera.update(2.0, 1.0, 2.0)["blocked"])

    def test_uncovering_the_lens_clears_the_state(self):
        camera = monitor()

        feed(camera, 2.0, 1.0, 0.0, 6.0)
        self.assertTrue(camera.blocked)

        result = camera.update(120.0, 45.0, 6.5)

        self.assertFalse(result["blocked"])
        self.assertEqual(result["blocked_duration"], 0.0)

    def test_the_clock_restarts_after_an_interruption(self):
        camera = monitor()

        feed(camera, 2.0, 1.0, 0.0, 1.5)
        camera.update(120.0, 45.0, 2.0)

        # Without a restart the earlier dark run would carry over and
        # this frame would already count as two seconds of cover.
        self.assertFalse(camera.update(2.0, 1.0, 2.5)["blocked"])

    def test_the_duration_is_reported_while_blocked(self):
        camera = monitor()

        result = feed(camera, 2.0, 1.0, 0.0, 5.0)

        self.assertAlmostEqual(result["blocked_duration"], 5.0)

    def test_the_readings_are_passed_through(self):
        result = monitor().update(42.0, 17.0, 0.0)

        self.assertEqual(result["brightness"], 42.0)
        self.assertEqual(result["variance"], 17.0)


def make_state(face_detected, camera=None):

    state = {"face_detected": face_detected}

    if camera is not None:
        state["camera"] = camera

    return state


def blocked_camera(blocked=True):
    return {
        "blocked": blocked,
        "brightness": 2.0,
        "variance": 1.0,
        "blocked_duration": 5.0 if blocked else 0.0
    }


def alerts(risk):
    return {alert["message"] for alert in risk["alerts"]}


class BlockedCameraScoringTest(unittest.TestCase):
    """
    What the driver is told. Both situations leave the pipeline with no
    face, so the only thing separating them is the camera reading.
    """

    def score(self, state, timestamp=10.0):
        index = DriverSafetyIndex()

        # One earlier frame so the face has been missing long enough
        # for DRIVER NOT VISIBLE to be on the table at all.
        index.update(state, 0.0)

        return index.update(state, timestamp)

    def test_a_covered_lens_reports_the_camera(self):
        risk = self.score(make_state(False, blocked_camera()))

        self.assertIn("CAMERA BLOCKED", alerts(risk))

    def test_a_covered_lens_does_not_also_report_a_missing_driver(self):
        """
        One cause, one alert. Sending the driver looking for a problem
        in the seat when the problem is the lens wastes the alert.
        """

        risk = self.score(make_state(False, blocked_camera()))

        self.assertNotIn("DRIVER NOT VISIBLE", alerts(risk))

    def test_an_empty_seat_still_reports_a_missing_driver(self):
        risk = self.score(make_state(False, blocked_camera(blocked=False)))

        self.assertIn("DRIVER NOT VISIBLE", alerts(risk))
        self.assertNotIn("CAMERA BLOCKED", alerts(risk))

    def test_a_cover_over_a_present_driver_still_reports_the_camera(self):
        """
        The laptop case: the shutter is closed, the driver never left.
        """

        risk = self.score(make_state(False, blocked_camera()))

        self.assertIn("camera_blocked", risk["penalties"])

    def test_a_blocked_camera_costs_points(self):
        risk = self.score(make_state(False, blocked_camera()))

        self.assertLess(risk["score"], 100.0)

    def test_a_state_without_a_camera_reading_behaves_as_before(self):
        """
        Offline callers and the older tests pass no camera key; they
        should still get the absent-driver verdict.
        """

        risk = self.score(make_state(False))

        self.assertIn("DRIVER NOT VISIBLE", alerts(risk))
        self.assertNotIn("CAMERA BLOCKED", alerts(risk))


class BlockedSpellCountTest(unittest.TestCase):
    """
    One covered-lens spell counts once, so the session summary can say
    how many times the camera was defeated rather than for how many
    frames.
    """

    def test_it_starts_at_zero(self):
        self.assertEqual(monitor().blocked_count, 0)

    def test_a_long_spell_counts_once(self):
        result = feed(monitor(), 2.0, 1.0, 0.0, 60.0)

        self.assertTrue(result["blocked"])
        self.assertEqual(result["blocked_count"], 1)

    def test_two_spells_count_twice(self):
        camera = monitor()

        feed(camera, 2.0, 1.0, 0.0, 10.0)
        feed(camera, 120.0, 45.0, 10.5, 15.0)
        result = feed(camera, 2.0, 1.0, 15.5, 25.0)

        self.assertEqual(result["blocked_count"], 2)

    def test_a_brief_dip_counts_nothing(self):
        """Shorter than the two seconds the monitor insists on."""

        result = feed(monitor(), 2.0, 1.0, 0.0, 1.0)

        self.assertEqual(result["blocked_count"], 0)

    def test_a_dark_cabin_counts_nothing(self):
        """Dark but not flat: shapes and sensor noise are still there."""

        result = feed(monitor(), 12.0, 25.0, 0.0, 60.0)

        self.assertEqual(result["blocked_count"], 0)


if __name__ == "__main__":
    unittest.main()

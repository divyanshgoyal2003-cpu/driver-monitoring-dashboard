"""
Hands on the wheel.

Most of these are about the monitor's right to complain at all. The
camera is aimed at a face, so on many setups the wheel is out of
shot and "no hands found" is simply the normal state. A monitor that
read that as hands off the wheel would cry wolf for an entire drive,
which is worse than saying nothing.
"""

import unittest

from analytics.wheel_monitor import WheelMonitor
from risk_engine.scoring import DriverSafetyIndex
from risk_engine import weights
from app.config import RISK_PENALTIES


def seen(count):
    return {"count": count, "confidence": 0.9, "hands": []}


def monitor():
    return WheelMonitor(off_duration=3.0, window=60.0, min_sightings=4)


def feed(wheel, count, start, end, step=0.5):
    """Hold a hand count from `start` to `end`; returns the last result."""

    result = None
    timestamp = start

    while timestamp <= end + 1e-9:
        result = wheel.update(seen(count), timestamp)
        timestamp += step

    return result


class CanSeeTest(unittest.TestCase):
    """Whether an absence of hands is evidence of anything."""

    def test_it_starts_unable_to_see(self):
        wheel = monitor()

        self.assertFalse(wheel.can_see)
        self.assertEqual(wheel.update(seen(0), 0.0)["status"], "UNKNOWN")

    def test_a_camera_that_never_sees_hands_never_complains(self):
        """
        The case that matters most: a wheel below the bottom edge of
        the frame. Ten minutes of no hands is not ten minutes of
        hands off the wheel.
        """

        wheel = monitor()
        result = feed(wheel, 0, 0.0, 600.0, step=5.0)

        self.assertFalse(wheel.can_see)
        self.assertEqual(result["status"], "UNKNOWN")
        self.assertFalse(result["hands_off"])

    def test_seeing_hands_often_enough_earns_the_right_to_judge(self):
        wheel = monitor()

        feed(wheel, 2, 0.0, 2.0)

        self.assertTrue(wheel.can_see)

    def test_sightings_expire_so_a_moved_camera_stops_claiming(self):
        """
        A camera knocked out of position should fall back to UNKNOWN
        rather than reporting the hands it can no longer see as gone.
        """

        wheel = monitor()

        feed(wheel, 2, 0.0, 2.0)
        self.assertTrue(wheel.can_see)

        result = feed(wheel, 0, 70.0, 140.0, step=5.0)

        self.assertFalse(wheel.can_see)
        self.assertEqual(result["status"], "UNKNOWN")


class StatusTest(unittest.TestCase):

    def ready(self):
        """A monitor that has proved it can see the wheel."""

        wheel = monitor()
        feed(wheel, 2, 0.0, 2.0)

        return wheel

    def test_two_hands_is_on_the_wheel(self):
        self.assertEqual(self.ready().update(seen(2), 3.0)["status"], "BOTH")

    def test_one_hand_is_reported_but_never_alarms(self):
        """
        Changing gear and indicating are one-handed and legal. A
        monitor that complained would be switched off.
        """

        result = feed(self.ready(), 1, 3.0, 60.0)

        self.assertEqual(result["status"], "ONE")
        self.assertFalse(result["hands_off"])

    def test_hands_gone_long_enough_is_hands_off(self):
        result = feed(self.ready(), 0, 3.0, 10.0)

        self.assertEqual(result["status"], "NONE")
        self.assertTrue(result["hands_off"])

    def test_hands_gone_briefly_is_not(self):
        """Long enough to sit out a gear change."""

        result = feed(self.ready(), 0, 3.0, 5.0)

        self.assertFalse(result["hands_off"])

    def test_the_clock_restarts_when_a_hand_comes_back(self):
        wheel = self.ready()

        feed(wheel, 0, 3.0, 5.0)
        wheel.update(seen(2), 5.5)

        result = feed(wheel, 0, 6.0, 8.0)

        self.assertFalse(result["hands_off"])

    def test_the_duration_is_reported(self):
        result = feed(self.ready(), 0, 3.0, 11.0)

        self.assertGreaterEqual(result["missing_for"], 7.9)

    def test_frames_without_an_observation_hold_the_state(self):
        """
        The detector runs on one frame in six; the rest pass None.
        """

        wheel = self.ready()

        feed(wheel, 0, 3.0, 10.0)
        self.assertTrue(wheel.update(None, 10.1)["hands_off"])

        wheel.update(seen(2), 10.2)
        self.assertFalse(wheel.update(None, 10.3)["hands_off"])


class HandsOffScoringTest(unittest.TestCase):

    def state(self, wheel):
        return {"face_detected": True, "wheel": wheel}

    def wheel(self, status, missing_for=0.0):
        return {
            "status": status,
            "count": 0 if status == "NONE" else 2,
            "hands_off": status == "NONE",
            "missing_for": missing_for,
            "can_see": status != "UNKNOWN"
        }

    def score(self, wheel):
        return DriverSafetyIndex().update(self.state(wheel), 0.0)

    def messages(self, risk):
        return {alert["message"] for alert in risk["alerts"]}

    def test_hands_off_raises_an_alert(self):
        risk = self.score(self.wheel("NONE", 5.0))

        self.assertIn("HANDS OFF WHEEL", self.messages(risk))
        self.assertLess(risk["score"], 100.0)

    def test_hands_on_does_not(self):
        risk = self.score(self.wheel("BOTH"))

        self.assertNotIn("HANDS OFF WHEEL", self.messages(risk))

    def test_one_hand_does_not(self):
        risk = self.score(self.wheel("ONE"))

        self.assertNotIn("HANDS OFF WHEEL", self.messages(risk))

    def test_a_camera_that_cannot_see_the_wheel_never_alerts(self):
        risk = self.score(self.wheel("UNKNOWN"))

        self.assertNotIn("HANDS OFF WHEEL", self.messages(risk))

    def test_a_state_without_a_wheel_reading_is_fine(self):
        """Offline callers and older tests pass no wheel key."""

        risk = DriverSafetyIndex().update({"face_detected": True}, 0.0)

        self.assertNotIn("HANDS OFF WHEEL", self.messages(risk))

    def test_longer_off_the_wheel_costs_more(self):
        brief = self.score(self.wheel("NONE", 3.0))["penalties"]["hands_off"]
        long = self.score(self.wheel("NONE", 12.0))["penalties"]["hands_off"]

        self.assertLess(brief, long)
        self.assertAlmostEqual(long, RISK_PENALTIES["hands_off"])

    def test_severity_needs_the_duration_to_scale(self):
        without = weights.severity("hands_off", {"wheel": {}})

        self.assertEqual(without, 1.0)


class BlockedLensTest(unittest.TestCase):
    """
    A covered lens is a wheel out of shot arrived at differently.

    This was found by running the harness over a recording, not by a
    test: the clip covered the lens for eight seconds and the report
    came back with a hands-off episode nobody had performed.
    """

    def ready(self):
        wheel = monitor()
        feed(wheel, 2, 0.0, 2.0)

        return wheel

    def blind(self, wheel, start, end, step=0.5):
        result = None
        timestamp = start

        while timestamp <= end + 1e-9:
            result = wheel.update(None, timestamp, blocked=True)
            timestamp += step

        return result

    def test_a_blocked_lens_does_not_report_hands_off(self):
        result = self.blind(self.ready(), 3.0, 20.0)

        self.assertEqual(result["status"], "UNKNOWN")
        self.assertFalse(result["hands_off"])

    def test_a_blocked_lens_counts_no_episodes(self):
        wheel = self.ready()

        self.blind(wheel, 3.0, 40.0)

        self.assertEqual(wheel.hands_off_count, 0)

    def test_the_clock_does_not_run_while_blind(self):
        """
        Otherwise the moment the lens is uncovered the monitor has
        several seconds of accumulated absence in hand and alarms
        before the detector has looked even once.
        """

        wheel = self.ready()

        self.blind(wheel, 3.0, 30.0)
        result = wheel.update(None, 30.5)

        self.assertAlmostEqual(result["missing_for"], 0.0)
        self.assertFalse(result["hands_off"])

    def test_a_brief_block_does_not_cost_the_right_to_judge(self):
        """
        Sightings expire on their own schedule; they are not thrown
        away because a hand passed over the lens.
        """

        wheel = self.ready()

        self.blind(wheel, 3.0, 6.0)

        self.assertTrue(wheel.can_see)

    def test_it_judges_again_once_the_lens_is_clear(self):
        wheel = self.ready()

        self.blind(wheel, 3.0, 10.0)
        result = feed(wheel, 0, 10.5, 20.0)

        self.assertEqual(result["status"], "NONE")
        self.assertTrue(result["hands_off"])

    def test_a_blocked_lens_while_hands_were_off_stops_claiming(self):
        """
        The episode was real until the view was lost. Once it is lost
        the monitor has nothing to say about it.
        """

        wheel = self.ready()

        running = feed(wheel, 0, 3.0, 10.0)
        self.assertTrue(running["hands_off"])

        result = self.blind(wheel, 10.5, 14.0)

        self.assertFalse(result["hands_off"])


class EpisodeCountTest(unittest.TestCase):
    """
    One spell of hands off the wheel counts once, however long it
    lasts, so the session summary reads like blinks and yawns rather
    than like a frame counter.
    """

    def ready(self):
        wheel = monitor()
        feed(wheel, 2, 0.0, 2.0)

        return wheel

    def test_it_starts_at_zero(self):
        self.assertEqual(monitor().hands_off_count, 0)

    def test_a_long_episode_counts_once(self):
        wheel = self.ready()

        result = feed(wheel, 0, 3.0, 40.0)

        self.assertTrue(result["hands_off"])
        self.assertEqual(result["hands_off_count"], 1)

    def test_two_episodes_count_twice(self):
        wheel = self.ready()

        feed(wheel, 0, 3.0, 10.0)
        feed(wheel, 2, 10.5, 12.0)
        result = feed(wheel, 0, 12.5, 20.0)

        self.assertEqual(result["hands_off_count"], 2)

    def test_hands_leaving_briefly_counts_nothing(self):
        wheel = self.ready()

        result = feed(wheel, 0, 3.0, 5.0)

        self.assertEqual(result["hands_off_count"], 0)

    def test_a_camera_that_cannot_see_counts_nothing(self):
        """
        The whole point of the can_see gate: a wheel out of shot must
        not fill the session summary with episodes that never
        happened.
        """

        wheel = monitor()

        result = feed(wheel, 0, 0.0, 600.0, step=5.0)

        self.assertEqual(result["hands_off_count"], 0)


if __name__ == "__main__":
    unittest.main()

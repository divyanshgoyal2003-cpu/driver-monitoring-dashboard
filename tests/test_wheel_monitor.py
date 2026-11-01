"""
Hands on the wheel.

Most of these are about the monitor's right to complain at all. The
camera is aimed at a face, so on many setups the wheel is out of
shot and "no hands found" is simply the normal state. A monitor that
read that as hands off the wheel would cry wolf for an entire drive,
which is worse than saying nothing.
"""

import unittest
from types import SimpleNamespace

from analytics.wheel_monitor import WheelMonitor
from risk_engine.scoring import DriverSafetyIndex
from risk_engine import weights
from app.config import RISK_PENALTIES


def seen(count):
    return {"count": count, "confidence": 0.9, "hands": []}


def monitor():
    return WheelMonitor(off_duration=3.0, window=60.0, min_share=0.3,
                        warmup=10.0)


def prove(wheel):
    """
    Give the monitor enough evidence that the wheel is in shot.

    The proving stretch runs up to t=0, so that each test's own
    timestamps still start from zero and read as they always did.
    """

    feed(wheel, 2, -12.0, 0.0)

    return wheel


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

        feed(wheel, 2, -12.0, 0.0)

        self.assertTrue(wheel.can_see)

    def test_sightings_expire_so_a_moved_camera_stops_claiming(self):
        """
        A camera knocked out of position should fall back to UNKNOWN
        rather than reporting the hands it can no longer see as gone.
        """

        wheel = monitor()

        prove(wheel)
        self.assertTrue(wheel.can_see)

        result = feed(wheel, 0, 70.0, 140.0, step=5.0)

        self.assertFalse(wheel.can_see)
        self.assertEqual(result["status"], "UNKNOWN")


class ShareOfTheWindowTest(unittest.TestCase):
    """
    How much evidence the wheel being in shot takes.

    A count was the first answer and it was far too cheap. Hands are
    looked for on one frame in six, so ten sightings is two seconds of
    one continuous sighting -- and two seconds bought a whole minute of
    judgement. Measured on a recording: the driver held a phone up
    beside their face, the monitor decided 1.7 seconds into that it
    could see the wheel, and it then called the next thirty seconds of
    hands resting in a lap hands off the wheel. 1520 frames reported
    against 349 performed.
    """

    def test_a_brief_sighting_does_not_prove_the_wheel_is_in_shot(self):
        """The recording's false positive, at the root."""

        wheel = monitor()

        feed(wheel, 2, 0.0, 2.0)
        result = feed(wheel, 0, 2.5, 40.0)

        self.assertFalse(wheel.can_see)
        self.assertEqual(result["status"], "UNKNOWN")
        self.assertEqual(result["hands_off_count"], 0)

    def test_hands_there_most_of_the_time_does(self):
        """A wheel in shot has hands on it nearly always."""

        wheel = monitor()

        result = feed(wheel, 2, 0.0, 40.0)

        self.assertTrue(wheel.can_see)
        self.assertEqual(result["status"], "BOTH")

    def test_nothing_is_claimed_before_the_window_has_filled(self):
        """
        Otherwise the first hand of the drive is 100% of the evidence
        there is.
        """

        wheel = monitor()

        feed(wheel, 2, 0.0, 5.0)

        self.assertFalse(wheel.can_see)

    def test_a_long_absence_does_not_switch_the_alarm_off(self):
        """
        The trap in measuring a share: a genuine spell of hands off the
        wheel drives the share down, and a monitor that read that as
        proof it could not see the wheel would give up precisely when
        the violation got serious.
        """

        wheel = prove(monitor())

        result = feed(wheel, 0, 0.5, 50.0)

        self.assertTrue(wheel.can_see)
        self.assertTrue(result["hands_off"])

    def test_a_whole_window_of_nothing_does_give_up_the_view(self):
        """
        Which is the different thing: a camera that has been moved, or
        one that never had the wheel in frame.
        """

        wheel = prove(monitor())

        feed(wheel, 0, 0.5, 90.0, step=2.0)

        self.assertFalse(wheel.can_see)

    def test_the_view_has_to_be_earned_again_afterwards(self):
        """
        A single hand after a blackout must not re-open judgement, or
        the cheap unlock is back by another route.
        """

        wheel = prove(monitor())

        feed(wheel, 0, 0.5, 90.0, step=2.0)
        self.assertFalse(wheel.can_see)

        feed(wheel, 2, 92.0, 94.0)

        self.assertFalse(wheel.can_see)


class StatusTest(unittest.TestCase):

    def ready(self):
        """A monitor that has proved it can see the wheel."""

        return prove(monitor())

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
        return prove(monitor())

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
        return prove(monitor())

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


def hand_at(y):
    """A hand whose landmarks all sit at normalized height y."""

    return [SimpleNamespace(x=0.5, y=y, z=0.0) for _ in range(21)]


def hands_at(*heights):
    return {
        "count": len(heights),
        "confidence": 0.9,
        "hands": [hand_at(y) for y in heights]
    }


class WheelRegionTest(unittest.TestCase):
    """
    Seeing a hand is not the same as seeing the wheel.

    Found by running the harness over a recording. The driver held a
    phone beside their face for eight seconds; that was enough hand
    sightings for the monitor to decide the wheel was in shot, and it
    then called the next thirty seconds -- hands resting in a lap,
    out of frame -- hands off the wheel. Three episodes reported, one
    performed, 17% precision.

    CHIN is the line: a wheel is below the driver's face, so hands
    above it say nothing either way.
    """

    LINE = 0.66

    def feed(self, wheel, observation, start, end, step=0.5):
        result = None
        timestamp = start

        while timestamp <= end + 1e-9:
            result = wheel.update(observation, timestamp,
                                  wheel_line=self.LINE)
            timestamp += step

        return result

    def test_hands_by_the_face_do_not_earn_the_right_to_judge(self):
        wheel = monitor()

        self.feed(wheel, hands_at(0.3, 0.35), 0.0, 20.0)

        self.assertFalse(wheel.can_see)

    def test_hands_at_the_wheel_do(self):
        wheel = monitor()

        self.feed(wheel, hands_at(0.8, 0.85), 0.0, 20.0)

        self.assertTrue(wheel.can_see)

    def test_a_phone_by_the_face_does_not_start_an_episode(self):
        """The recording's false positive, in miniature."""

        wheel = monitor()

        self.feed(wheel, hands_at(0.3), 0.0, 10.0)
        result = self.feed(wheel, hands_at(), 10.5, 60.0)

        self.assertEqual(result["status"], "UNKNOWN")
        self.assertFalse(result["hands_off"])
        self.assertEqual(result["hands_off_count"], 0)

    def test_a_raised_hand_does_not_silence_a_real_alarm(self):
        """
        The same confusion the other way, and the more dangerous one:
        a hand lifted to the face must not read as a hand on the
        wheel.
        """

        wheel = monitor()

        self.feed(wheel, hands_at(0.8, 0.85), 0.0, 10.0)
        result = self.feed(wheel, hands_at(0.3), 10.5, 30.0)

        self.assertEqual(result["status"], "NONE")
        self.assertTrue(result["hands_off"])

    def test_one_hand_up_and_one_on_the_wheel_is_one_hand(self):
        wheel = monitor()

        result = self.feed(wheel, hands_at(0.3, 0.8), 0.0, 20.0)

        self.assertEqual(result["status"], "ONE")
        self.assertFalse(result["hands_off"])

    def test_without_a_line_every_hand_still_counts(self):
        """
        Callers with no face detector, and every test above this one.
        """

        wheel = monitor()
        timestamp = 0.0

        while timestamp <= 20.0:
            result = wheel.update(hands_at(0.3, 0.35), timestamp)
            timestamp += 0.5

        self.assertTrue(wheel.can_see)
        self.assertEqual(result["status"], "BOTH")


if __name__ == "__main__":
    unittest.main()

import unittest
from unittest import mock

from utils.alert_sound import AlertSound
from app.config import FATIGUE_CONDITIONS


class ImmediateThread:
    """Stands in for threading.Thread and runs the target on start()."""

    def __init__(self, target, args=(), daemon=None):
        self.target = target
        self.args = args

    def start(self):
        self.target(*self.args)


def alert(name, severity, message="..."):
    return {"name": name, "message": message, "severity": severity}


PHONE = [alert("phone", "high", "PHONE IN USE")]
MICROSLEEP = [alert("microsleep", "critical", "MICROSLEEP - WAKE UP")]
DROWSY = [alert("drowsy", "high", "DROWSINESS DETECTED")]
TIRED = [alert("tired", "medium", "SIGNS OF FATIGUE")]


class PatternChoiceTest(unittest.TestCase):
    """
    Which pattern a situation earns, before any timing is involved.
    """

    def test_nothing_worth_sounding(self):
        self.assertIsNone(AlertSound.pattern_for([]))
        self.assertIsNone(AlertSound.pattern_for(TIRED))

    def test_a_microsleep_rouses(self):
        self.assertEqual(
            AlertSound.pattern_for(MICROSLEEP), ("rouse", "critical")
        )

    def test_a_phone_only_notifies(self):
        self.assertEqual(AlertSound.pattern_for(PHONE), ("notify", "high"))

    def test_drowsiness_rouses_even_though_it_is_only_high(self):
        """
        The point of the split. Drowsy and phone share a severity, but
        a phone reminder only has to be noticed and a fatigue alert has
        to reach someone whose eyes are closing.
        """

        self.assertEqual(AlertSound.pattern_for(DROWSY), ("rouse", "high"))

    def test_fatigue_wins_against_the_same_severity(self):
        self.assertEqual(
            AlertSound.pattern_for(PHONE + DROWSY), ("rouse", "high")
        )

    def test_critical_is_considered_before_high(self):
        self.assertEqual(
            AlertSound.pattern_for(PHONE + MICROSLEEP),
            ("rouse", "critical")
        )

    def test_every_fatigue_condition_rouses(self):
        for name in FATIGUE_CONDITIONS:
            self.assertEqual(
                AlertSound.pattern_for([alert(name, "high")]),
                ("rouse", "high"),
                f"{name} should rouse"
            )

    def test_an_alert_without_a_name_still_sounds(self):
        """Older callers passed message and severity only."""

        self.assertEqual(
            AlertSound.pattern_for([{"severity": "high"}]),
            ("notify", "high")
        )


class PatternShapeTest(unittest.TestCase):

    @staticmethod
    def seconds(pattern):
        return sum(duration + gap for _f, duration, gap in pattern) / 1000.0

    def test_the_rousing_pattern_is_the_longer_one(self):
        rouse = self.seconds(AlertSound.PATTERNS["rouse"])
        notify = self.seconds(AlertSound.PATTERNS["notify"])

        self.assertGreater(rouse, notify)
        self.assertGreater(rouse, 1.5)

    def test_the_rousing_pattern_pulses_rather_than_holding_one_note(self):
        """
        A steady tone is the easy thing to sleep through. Interruption
        and a rising pitch are what read as urgent.
        """

        pattern = AlertSound.PATTERNS["rouse"]
        pitches = [frequency for frequency, _d, _g in pattern]

        self.assertGreater(len(pattern), 2)
        self.assertGreater(len(set(pitches)), 1)
        self.assertGreater(max(pitches), min(pitches))

    def test_patterns_are_well_formed(self):
        for name, pattern in AlertSound.PATTERNS.items():
            for frequency, duration, gap in pattern:
                # winsound.Beep rejects anything outside 37-32767 Hz.
                self.assertGreaterEqual(frequency, 37, name)
                self.assertLessEqual(frequency, 32767, name)
                self.assertGreater(duration, 0, name)
                self.assertGreaterEqual(gap, 0, name)

    def test_each_cooldown_outlasts_its_pattern(self):
        """
        Otherwise an alert restarts the moment it ends and the silence
        between - the thing that stops it becoming background - is
        gone.
        """

        sound = AlertSound()

        self.assertGreater(
            sound.cooldowns["critical"], self.seconds(AlertSound.PATTERNS["rouse"])
        )
        self.assertGreater(
            sound.cooldowns["high"], self.seconds(AlertSound.PATTERNS["rouse"])
        )


class PlaybackTest(unittest.TestCase):

    def setUp(self):
        self.sound = AlertSound(enabled=True, critical_cooldown=3.0,
                                high_cooldown=4.0)
        self.sound.available = True
        self.played = []

        patcher = mock.patch("utils.alert_sound.threading.Thread",
                             ImmediateThread)
        patcher.start()
        self.addCleanup(patcher.stop)

        self.sound._play = self._record

    def _record(self, pattern):
        self.played.append(pattern)
        self.sound.playing = False

    def at(self, seconds):
        return mock.patch("utils.alert_sound.time.monotonic",
                          return_value=seconds)

    def test_no_sound_for_medium_alerts(self):
        with self.at(0.0):
            self.sound.update(TIRED)

        self.assertEqual(self.played, [])

    def test_a_phone_alert_repeats_on_its_cooldown(self):
        for t in (0.0, 1.0, 3.0, 4.5):
            with self.at(t):
                self.sound.update(PHONE)

        self.assertEqual(
            self.played, [AlertSound.PATTERNS["notify"]] * 2
        )

    def test_a_microsleep_repeats_on_the_shorter_cooldown(self):
        for t in (0.0, 1.5, 3.2):
            with self.at(t):
                self.sound.update(MICROSLEEP + PHONE)

        self.assertEqual(self.played, [AlertSound.PATTERNS["rouse"]] * 2)

    def test_drowsiness_plays_the_rousing_pattern(self):
        with self.at(0.0):
            self.sound.update(DROWSY)

        self.assertEqual(self.played, [AlertSound.PATTERNS["rouse"]])

    def test_nothing_starts_while_one_is_still_playing(self):
        self.sound.playing = True

        with self.at(100.0):
            self.sound.update(MICROSLEEP)

        self.assertEqual(self.played, [])

    def test_muted(self):
        self.assertFalse(self.sound.toggle())

        with self.at(0.0):
            self.sound.update(MICROSLEEP)

        self.assertEqual(self.played, [])

    def test_muting_cuts_a_pattern_short(self):
        """
        A two second alarm you cannot interrupt is its own problem.
        """

        self.sound.toggle()

        self.assertTrue(self.sound._silenced.is_set())

        self.sound.toggle()

        self.assertFalse(self.sound._silenced.is_set())

    def test_silent_without_winsound(self):
        self.sound.available = False

        with self.at(0.0):
            self.sound.update(MICROSLEEP)

        self.assertEqual(self.played, [])


if __name__ == "__main__":
    unittest.main()

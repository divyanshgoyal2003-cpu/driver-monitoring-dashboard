import unittest
from unittest import mock

from utils.alert_sound import AlertSound


class ImmediateThread:
    """Stands in for threading.Thread and runs the target on start()."""

    def __init__(self, target, args=(), daemon=None):
        self.target = target
        self.args = args

    def start(self):
        self.target(*self.args)


HIGH = [{"message": "PHONE IN USE", "severity": "high"}]
CRITICAL = [{"message": "MICROSLEEP - WAKE UP", "severity": "critical"}]
MEDIUM = [{"message": "SIGNS OF FATIGUE", "severity": "medium"}]


class AlertSoundTest(unittest.TestCase):

    def setUp(self):
        self.sound = AlertSound(enabled=True, critical_cooldown=1.0,
                                high_cooldown=3.0)
        self.sound.available = True
        self.beeps = []

        patcher = mock.patch("utils.alert_sound.threading.Thread",
                             ImmediateThread)
        patcher.start()
        self.addCleanup(patcher.stop)

        self.sound._beep = self._record

    def _record(self, frequency, duration):
        self.beeps.append((frequency, duration))
        self.sound.playing = False

    def at(self, seconds):
        return mock.patch("utils.alert_sound.time.monotonic",
                          return_value=seconds)

    def test_no_sound_for_medium_alerts(self):
        with self.at(0.0):
            self.sound.update(MEDIUM)
        self.assertEqual(self.beeps, [])

    def test_high_alert_beeps_with_cooldown(self):
        for t in (0.0, 1.0, 2.0, 3.5):
            with self.at(t):
                self.sound.update(HIGH)

        self.assertEqual(self.beeps, [AlertSound.TONES["high"]] * 2)

    def test_critical_alert_uses_its_own_tone_and_cooldown(self):
        for t in (0.0, 0.5, 1.2):
            with self.at(t):
                self.sound.update(CRITICAL + HIGH)

        self.assertEqual(self.beeps, [AlertSound.TONES["critical"]] * 2)

    def test_muted(self):
        self.assertFalse(self.sound.toggle())

        with self.at(0.0):
            self.sound.update(CRITICAL)

        self.assertEqual(self.beeps, [])

    def test_silent_without_winsound(self):
        self.sound.available = False

        with self.at(0.0):
            self.sound.update(CRITICAL)

        self.assertEqual(self.beeps, [])


if __name__ == "__main__":
    unittest.main()

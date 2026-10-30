import sys
import threading
import time

from app.config import (
    ENABLE_SOUND_ALERTS,
    CRITICAL_ALERT_COOLDOWN,
    HIGH_ALERT_COOLDOWN,
    ALERT_TONE_ROUSE,
    ALERT_TONE_NOTIFY,
    FATIGUE_CONDITIONS
)


class AlertSound:
    """
    Sounds a pattern while a critical or high-severity alert is active.

    Two patterns, chosen by what is wrong rather than by severity
    alone. A drowsy driver and a driver holding a phone are both
    "high", but a phone reminder only has to be noticed, while a
    fatigue alert has to reach someone whose eyes have already closed -
    so the fatigue conditions get the longer rising pattern.

    Uses winsound, so it only makes a sound on Windows; elsewhere it
    stays silent. Patterns play on a background thread, so the video
    never stalls waiting for a two-second alarm, and pressing S cuts
    one off part-way rather than waiting for it to finish.
    """

    PATTERNS = {
        "rouse": ALERT_TONE_ROUSE,
        "notify": ALERT_TONE_NOTIFY,
    }

    def __init__(self, enabled=ENABLE_SOUND_ALERTS,
                 critical_cooldown=CRITICAL_ALERT_COOLDOWN,
                 high_cooldown=HIGH_ALERT_COOLDOWN):

        self.available = sys.platform == "win32"
        self.enabled = enabled
        self.cooldowns = {
            "critical": critical_cooldown,
            "high": high_cooldown,
        }

        self.last_played = None
        self.playing = False

        # Set while sound is off, to cut a pattern short mid-way.
        self._silenced = threading.Event()

    def toggle(self):

        self.enabled = not self.enabled

        if self.enabled:
            self._silenced.clear()
        else:
            self._silenced.set()

        return self.enabled

    @classmethod
    def pattern_for(cls, alerts):
        """
        ("rouse" or "notify", severity) for the alert worth sounding, or
        None when none of them is.

        Fatigue wins over anything else of the same severity: of the
        two, it is the one the driver cannot simply decide to stop
        doing.
        """

        loudest = None

        for severity in ("critical", "high"):
            matching = [a for a in alerts if a.get("severity") == severity]

            if not matching:
                continue

            fatigue = any(
                a.get("name") in FATIGUE_CONDITIONS for a in matching
            )

            loudest = ("rouse" if fatigue else "notify", severity)
            break

        return loudest

    def update(self, alerts):

        if not (self.enabled and self.available) or self.playing:
            return

        choice = self.pattern_for(alerts)

        if choice is None:
            return

        voice, severity = choice

        now = time.monotonic()

        if (self.last_played is not None
                and now - self.last_played < self.cooldowns[severity]):
            return

        self.last_played = now
        self.playing = True

        threading.Thread(
            target=self._play,
            args=(self.PATTERNS[voice],),
            daemon=True
        ).start()

    def _play(self, pattern):

        try:
            import winsound

            for frequency, duration, gap in pattern:

                if self._silenced.is_set():
                    break

                winsound.Beep(frequency, duration)

                if gap:
                    # Event.wait rather than sleep: muting cuts the
                    # gap short too, so S stops the alarm at once.
                    self._silenced.wait(gap / 1000.0)

        except Exception:
            # No sound device: alerts stay visual only.
            pass

        finally:
            self.playing = False

import sys
import threading
import time

from app.config import (
    ENABLE_SOUND_ALERTS,
    CRITICAL_ALERT_COOLDOWN,
    HIGH_ALERT_COOLDOWN
)


class AlertSound:
    """
    Beeps while a critical or high-severity alert is active.

    Uses winsound, so it only makes a sound on Windows; elsewhere it stays
    silent. Each beep plays on a background thread so the video never
    stalls.
    """

    # severity -> (frequency Hz, duration ms)
    TONES = {
        "critical": (1500, 600),
        "high": (1000, 300),
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

    def toggle(self):
        self.enabled = not self.enabled
        return self.enabled

    def update(self, alerts):

        if not (self.enabled and self.available) or self.playing:
            return

        severities = {alert["severity"] for alert in alerts}

        if "critical" in severities:
            severity = "critical"
        elif "high" in severities:
            severity = "high"
        else:
            return

        now = time.monotonic()

        if (self.last_played is not None
                and now - self.last_played < self.cooldowns[severity]):
            return

        self.last_played = now
        self.playing = True

        threading.Thread(
            target=self._beep,
            args=self.TONES[severity],
            daemon=True
        ).start()

    def _beep(self, frequency, duration):

        try:
            import winsound
            winsound.Beep(frequency, duration)

        except Exception:
            # No sound device: alerts stay visual only.
            pass

        finally:
            self.playing = False

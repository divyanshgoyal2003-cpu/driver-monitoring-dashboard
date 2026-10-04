from app.config import PHONE_MIN_DURATION, PHONE_RELEASE_TIME


class PhoneUsageDetector:
    """
    Turns per-frame phone detections into a steady "phone in use" state.

    Usage starts once phones have been detected over a span of at least
    PHONE_MIN_DURATION seconds, and ends once no phone has been seen for
    PHONE_RELEASE_TIME seconds.
    """

    def __init__(self, min_duration=PHONE_MIN_DURATION,
                 release_time=PHONE_RELEASE_TIME):

        self.min_duration = min_duration
        self.release_time = release_time

        self.first_seen = None
        self.last_seen = None
        self.in_use = False
        self.usage_count = 0

    def update(self, phones, timestamp):

        detected = bool(phones)

        if detected:
            if self.first_seen is None:
                self.first_seen = timestamp

            self.last_seen = timestamp

        elif (self.last_seen is not None
              and timestamp - self.last_seen > self.release_time):
            self.first_seen = None
            self.last_seen = None

        in_use = (
            self.first_seen is not None
            and self.last_seen - self.first_seen >= self.min_duration
        )

        if in_use and not self.in_use:
            self.usage_count += 1

        self.in_use = in_use

        return {
            "detected": detected,
            "in_use": in_use,
            "usage_duration": (
                timestamp - self.first_seen if in_use else 0.0
            ),
            "usage_count": self.usage_count,
            "phones": phones
        }

from collections import deque

from app.config import (
    MICROSLEEP_DURATION,
    PERCLOS_WINDOW,
    PERCLOS_TIRED,
    PERCLOS_DROWSY,
    PERCLOS_MIN_DATA
)


class DrowsinessDetector:
    """
    Turns the per-frame eye state from BlinkDetector into drowsiness
    indicators:

    - PERCLOS: the share of time the eyes were closed over the last
      PERCLOS_WINDOW seconds, the standard measure of fatigue.
    - Microsleep: the eyes staying closed for MICROSLEEP_DURATION.

    State is one of AWAKE, TIRED, DROWSY or MICROSLEEP.
    """

    # A long gap between frames (a stall) is only counted up to this, so
    # one frame cannot dominate the window.
    MAX_SAMPLE_GAP = 0.5

    def __init__(self, microsleep_duration=MICROSLEEP_DURATION,
                 window=PERCLOS_WINDOW,
                 tired_threshold=PERCLOS_TIRED,
                 drowsy_threshold=PERCLOS_DROWSY,
                 min_data=PERCLOS_MIN_DATA):

        self.microsleep_duration = microsleep_duration
        self.window = window
        self.tired_threshold = tired_threshold
        self.drowsy_threshold = drowsy_threshold
        self.min_data = min_data

        # (timestamp, duration, eyes closed)
        self.samples = deque()
        self.total_time = 0.0
        self.closed_time = 0.0
        self.last_timestamp = None

        self.microsleep_active = False
        self.microsleep_count = 0

    def update(self, blink_data, timestamp):

        if blink_data is None:
            # No face, no eye data. The time spent without a face is not
            # counted either way.
            self.last_timestamp = None
            self.microsleep_active = False
            return None

        if self.last_timestamp is not None:
            duration = min(timestamp - self.last_timestamp,
                           self.MAX_SAMPLE_GAP)

            if duration > 0:
                self._add_sample(timestamp, duration,
                                 blink_data["eyes_closed"])

        self.last_timestamp = timestamp

        self._expire(timestamp)

        perclos = (
            self.closed_time / self.total_time
            if self.total_time > 0
            else 0.0
        )

        ready = self.total_time >= self.min_data

        microsleep = (
            blink_data["closed_duration"] >= self.microsleep_duration
        )

        if microsleep and not self.microsleep_active:
            self.microsleep_count += 1

        self.microsleep_active = microsleep

        if microsleep:
            state = "MICROSLEEP"
        elif ready and perclos >= self.drowsy_threshold:
            state = "DROWSY"
        elif ready and perclos >= self.tired_threshold:
            state = "TIRED"
        else:
            state = "AWAKE"

        return {
            "perclos": perclos,
            "perclos_ready": ready,
            "microsleep": microsleep,
            "microsleep_count": self.microsleep_count,
            "state": state
        }

    def _add_sample(self, timestamp, duration, closed):

        self.samples.append((timestamp, duration, closed))
        self.total_time += duration

        if closed:
            self.closed_time += duration

    def _expire(self, timestamp):

        while self.samples and timestamp - self.samples[0][0] > self.window:
            _, duration, closed = self.samples.popleft()
            self.total_time -= duration

            if closed:
                self.closed_time -= duration

        # Keep floating-point drift in the running sums from building up.
        if not self.samples:
            self.total_time = 0.0
            self.closed_time = 0.0
        else:
            self.total_time = max(self.total_time, 0.0)
            self.closed_time = min(max(self.closed_time, 0.0),
                                   self.total_time)

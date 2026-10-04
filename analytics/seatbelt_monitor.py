from collections import deque

from app.config import (
    SEATBELT_WINDOW,
    SEATBELT_ON_RATIO,
    SEATBELT_OFF_RATIO
)


class SeatbeltMonitor:
    """
    Pools per-frame seatbelt observations over a time window into a
    steady status:

    - ON: the belt was found in at least ON_RATIO of the recent frames
      where the chest was visible.
    - OFF: found in OFF_RATIO or fewer of them.
    - UNKNOWN: the chest was not visible in most recent frames.

    Between the two ratios the previous status holds, so a borderline
    scene does not flicker.
    """

    # Fewer visible frames than this in the window and no judgement is
    # made.
    MIN_VISIBLE_SAMPLES = 3

    def __init__(self, window=SEATBELT_WINDOW, on_ratio=SEATBELT_ON_RATIO,
                 off_ratio=SEATBELT_OFF_RATIO):

        self.window = window
        self.on_ratio = on_ratio
        self.off_ratio = off_ratio

        # (timestamp, chest visible, belt found)
        self.samples = deque()
        self.status = "UNKNOWN"
        self.last_observation = None

    def update(self, observation, timestamp):
        """
        observation: SeatbeltDetector.detect() output, or None when the
        detector did not run on this frame.
        """

        if observation is not None:
            self.samples.append((
                timestamp,
                observation["visible"],
                observation["belt_found"]
            ))
            self.last_observation = observation

        while self.samples and timestamp - self.samples[0][0] > self.window:
            self.samples.popleft()

        visible = [found for _, seen, found in self.samples if seen]

        if (len(visible) < self.MIN_VISIBLE_SAMPLES
                or len(visible) < len(self.samples) / 2):
            self.status = "UNKNOWN"
            ratio = 0.0

        else:
            ratio = sum(visible) / len(visible)

            if ratio >= self.on_ratio:
                self.status = "ON"
            elif ratio <= self.off_ratio:
                self.status = "OFF"
            elif self.status == "UNKNOWN":
                # No previous status to hold; lean to the safe side.
                self.status = "OFF"

        return {
            "status": self.status,
            "ratio": ratio,
            "observation": self.last_observation
        }

"""
Whether the driver's hands are on the wheel.

The honest difficulty here is framing, not detection. This camera is
aimed at a face: it sees the head and the upper chest, and on most
setups the wheel sits below the bottom edge. If it cannot see hands at
all, then "no hands found" is the normal state and alarming on it
would mean crying wolf for the whole drive.

So the monitor earns the right to complain. It watches for hands and
counts how often it finds them; only once it has seen them often
enough to prove the wheel is in shot does their absence start to mean
something. Until then the status is UNKNOWN, the same answer the
seatbelt detector gives when the chest is out of view. A monitor that
admits it cannot see is worth more than one that guesses.

One hand is reported but never alarmed on. Changing gear, indicating
and adjusting the heating are all one-handed and all legal; a system
that complained about them would be turned off within a week.
"""

from collections import deque

from app.config import (
    HANDS_OFF_TIME,
    HANDS_SEEN_WINDOW,
    HANDS_SEEN_SAMPLES
)


class WheelMonitor:

    def __init__(self, off_duration=HANDS_OFF_TIME,
                 window=HANDS_SEEN_WINDOW,
                 min_sightings=HANDS_SEEN_SAMPLES):

        self.off_duration = off_duration
        self.window = window
        self.min_sightings = min_sightings

        # (timestamp, hands found) for the recent past, used to decide
        # whether this camera can see hands at all.
        self.samples = deque()
        self.sightings = 0

        self.missing_since = None
        self.status = "UNKNOWN"
        self.count = 0

        # Episodes, not frames: one spell of hands off the wheel
        # counts once however long it lasts, so the session
        # summary reads the same way as blinks and yawns.
        self.hands_off_count = 0
        self.hands_off_active = False

    @property
    def can_see(self):
        """
        Whether an absence of hands is evidence of anything.

        True once hands have been found often enough in the recent past
        that the wheel is clearly within the camera's view.
        """

        return self.sightings >= self.min_sightings

    def update(self, observation, timestamp, blocked=False):
        """
        observation: MediaPipeHandLandmarker.detect() output, or None on
                     frames where the detector did not run.
        blocked:     whether the lens is covered.

        A covered lens is the same situation as a wheel out of shot,
        arrived at differently: the monitor cannot see, so an absence of
        hands is not evidence. Without this the clock would keep running
        while the camera was blind and the driver would be told to take
        hold of a wheel nobody could see -- charging one cause twice,
        which is exactly what suppressing DRIVER NOT VISIBLE while
        blocked already avoids.
        """

        if blocked:
            self.count = 0
            self.missing_since = None
            self.status = "UNKNOWN"
            self.hands_off_active = False

            # Sightings are left alone rather than cleared. They expire
            # on their own, and a lens covered for a moment should not
            # cost the monitor the right to judge once it is uncovered.
            self._forget_old(timestamp)

            return {
                "status": self.status,
                "count": 0,
                "hands_off": False,
                "hands_off_count": self.hands_off_count,
                "missing_for": 0.0,
                "can_see": self.can_see
            }

        if observation is not None:
            self.count = observation["count"]

            self.samples.append((timestamp, self.count > 0))

            if self.count > 0:
                self.sightings += 1
                self.missing_since = None

            elif self.missing_since is None:
                self.missing_since = timestamp

        self._forget_old(timestamp)

        missing_for = (
            0.0 if self.missing_since is None
            else max(timestamp - self.missing_since, 0.0)
        )

        if not self.can_see:
            self.status = "UNKNOWN"

        elif self.count >= 2:
            self.status = "BOTH"

        elif self.count == 1:
            self.status = "ONE"

        elif missing_for >= self.off_duration:
            self.status = "NONE"

        else:
            # Hands have just left the frame; not yet long enough to
            # call it.
            self.status = "ONE" if self.status == "ONE" else "BOTH"

        hands_off = self.status == "NONE"

        if hands_off and not self.hands_off_active:
            self.hands_off_count += 1

        self.hands_off_active = hands_off

        return {
            "status": self.status,
            "count": self.count,
            "hands_off": hands_off,
            "hands_off_count": self.hands_off_count,
            "missing_for": missing_for,
            "can_see": self.can_see
        }

    def _forget_old(self, timestamp):
        """
        Sightings are counted over a window, so a camera that is moved
        mid-drive stops claiming it can see the wheel.
        """

        while self.samples and timestamp - self.samples[0][0] > self.window:
            _, found = self.samples.popleft()

            if found:
                self.sightings -= 1

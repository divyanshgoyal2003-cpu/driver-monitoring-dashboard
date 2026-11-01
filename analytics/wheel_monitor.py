"""
Whether the driver's hands are on the wheel.

The honest difficulty here is framing, not detection. This camera is
aimed at a face: it sees the head and the upper chest, and on most
setups the wheel sits below the bottom edge. If it cannot see hands at
all, then "no hands found" is the normal state and alarming on it
would mean crying wolf for the whole drive.

So the monitor earns the right to complain. It watches for hands low
in the frame and measures what share of the recent past it found them
in; only once that share is high enough to prove the wheel is in shot
does their absence start to mean something. Until then the status is
UNKNOWN, the same answer the seatbelt detector gives when the chest is
out of view. A monitor that admits it cannot see is worth more than
one that guesses.

One hand is reported but never alarmed on. Changing gear, indicating
and adjusting the heating are all one-handed and all legal; a system
that complained about them would be turned off within a week.
"""

from collections import deque

from app.config import (
    HANDS_OFF_TIME,
    HANDS_SEEN_WINDOW,
    HANDS_SEEN_SHARE,
    HANDS_SEEN_WARMUP
)


class WheelMonitor:

    def __init__(self, off_duration=HANDS_OFF_TIME,
                 window=HANDS_SEEN_WINDOW,
                 min_share=HANDS_SEEN_SHARE,
                 warmup=HANDS_SEEN_WARMUP):

        self.off_duration = off_duration
        self.window = window
        self.min_share = min_share
        self.warmup = warmup

        # (timestamp, hands found) for the recent past, used to decide
        # whether this camera can see hands at all.
        self.samples = deque()
        self.sightings = 0

        # Whether the share rule has been met, which is a separate
        # question from whether hands are there now. See can_see.
        self.proved = False

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

        True once hands have been found in enough of the recent past
        that the wheel is clearly within the camera's view.
        """

        return self.proved

    def update(self, observation, timestamp, blocked=False,
               wheel_line=None):
        """
        observation: MediaPipeHandLandmarker.detect() output, or None on
                     frames where the detector did not run.
        blocked:     whether the lens is covered.
        wheel_line:  normalized y below which a hand is near the wheel
                     rather than near the face. The pipeline passes the
                     bottom of the driver's face box. None means the
                     driver has not been located, and then every hand
                     found is counted, which is what callers without a
                     face detector get.

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
            self._review_evidence(timestamp)

            return {
                "status": self.status,
                "count": 0,
                "hands_off": False,
                "hands_off_count": self.hands_off_count,
                "missing_for": 0.0,
                "can_see": self.can_see
            }

        if observation is not None:
            self.count = self._at_wheel(observation, wheel_line)

            self.samples.append((timestamp, self.count > 0))

            if self.count > 0:
                self.sightings += 1
                self.missing_since = None

            elif self.missing_since is None:
                self.missing_since = timestamp

        self._review_evidence(timestamp)

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

    @staticmethod
    def _at_wheel(observation, wheel_line):
        """
        How many of the hands found are low enough to be at a wheel.

        Seeing a hand is not the same as seeing the wheel, and the
        difference matters in both directions. Measured on a
        recording: the driver held a phone up beside their face for
        eight seconds, which was enough for the monitor to decide it
        could see the wheel, and it then reported the next
        thirty seconds -- hands resting in a lap, below the frame --
        as hands off the wheel. Three episodes were reported where one
        happened.

        The same confusion runs the other way, and is worse. A hand
        raised to the face would read as a hand on the wheel and
        silence a genuine alarm.

        A wheel is below the driver's chin, so hands above it are
        evidence of nothing either way.
        """

        if wheel_line is None:
            return observation["count"]

        hands = observation.get("hands") or []

        return sum(
            1 for hand in hands
            if hand and sum(point.y for point in hand) / len(hand)
            >= wheel_line
        )

    def _review_evidence(self, timestamp):
        """
        Expire old samples, then ask again whether the wheel is in shot.

        A share rather than a count, because a count is far cheaper to
        satisfy than it looks: hands are looked for on one frame in
        six, so ten sightings is two seconds of one continuous
        sighting. On a recording, a phone held up beside the driver's
        face bought sixty seconds of judgement after 1.7 seconds of
        being visible, and the thirty seconds of hands in a lap that
        followed were reported as hands off the wheel.

        The warmup is the other half of it: without it, the first hand
        of a drive is 100% of the evidence there is.

        Having been proved, the view is only given up when the whole
        window passes with no hands in it at all -- not merely when
        the share falls. The share falling is exactly what a long spell
        of hands off the wheel looks like from here, and a monitor that
        read its own alarm as proof that it could not see would switch
        itself off the longer the violation ran. A full minute with
        nothing is the different thing: a camera that has been moved,
        or one that never had the wheel in frame.
        """

        while self.samples and timestamp - self.samples[0][0] > self.window:
            _, found = self.samples.popleft()

            if found:
                self.sightings -= 1

        if not self.samples:
            return

        span = self.samples[-1][0] - self.samples[0][0]

        if span < self.warmup:
            return

        if self.sightings / len(self.samples) >= self.min_share:
            self.proved = True

        elif self.sightings == 0:
            self.proved = False

from app.config import (
    OFF_ROAD_YAW,
    OFF_ROAD_PITCH,
    DISTRACTION_DURATION,
    DISTRACTION_FACE_GAP,
    DISTRACTION_RECOVERY
)


class DistractionDetector:
    """
    Flags the driver as distracted when their gaze stays off the road for
    DISTRACTION_DURATION seconds.

    The horizontal direction uses the combined head + eye yaw from
    GazeEstimator when available (head yaw alone otherwise); the vertical
    direction uses head pitch.
    """

    def __init__(self, max_yaw=OFF_ROAD_YAW, max_pitch=OFF_ROAD_PITCH,
                 min_duration=DISTRACTION_DURATION,
                 face_gap=DISTRACTION_FACE_GAP,
                 recovery=DISTRACTION_RECOVERY):

        self.max_yaw = max_yaw
        self.max_pitch = max_pitch
        self.min_duration = min_duration
        self.face_gap = face_gap
        self.recovery = recovery

        self.off_road_since = None
        self.on_road_since = None
        self.last_seen = None
        self.distracted_active = False
        self.distraction_count = 0

    def update(self, head_pose, gaze, timestamp):

        if head_pose is None or not head_pose.get("reliable", True):
            # No face, or a face the pose estimator will not vouch for
            # because the mouth is open and it is no longer fitting a
            # rigid head. Either way there is no usable reading, and
            # nothing is reported -- the driver cannot be called
            # distracted on evidence nobody gathered.
            #
            # But the clock is held rather than cleared. Losing the
            # face is not the driver looking back at the road; it is
            # most often the landmarker failing on a head that is
            # turned far enough away to be hard to see, which is to
            # say it happens *because* they are looking away. Clearing
            # the clock made a long turn read as a string of short
            # ones, none of them lasting long enough to report. On a
            # recorded turn of nine seconds, two and a quarter were
            # lost to a face gap of two thirds of a second.
            #
            # The hold is bounded: past DISTRACTION_FACE_GAP there has
            # been no usable reading for too long to assume anything
            # about where the head was pointing, and an absent driver
            # is a separate condition with its own alert. A yawn runs
            # longer than that, so one does not preserve an episode
            # across itself -- it only avoids inventing one.
            self.distracted_active = False

            if (self.last_seen is not None
                    and timestamp - self.last_seen > self.face_gap):
                self.off_road_since = None

            return None

        self.last_seen = timestamp

        if not head_pose["calibrated"]:
            # The neutral pose is still being learned, so any offset
            # now is the camera's rather than the driver's.
            self.off_road_since = None
            self.on_road_since = None
            self.distracted_active = False
            return None

        yaw = gaze["gaze_yaw"] if gaze else head_pose["yaw"]
        pitch = head_pose["pitch"]

        off_road = abs(yaw) > self.max_yaw or abs(pitch) > self.max_pitch

        if off_road:
            self.on_road_since = None

            if self.off_road_since is None:
                self.off_road_since = timestamp

        elif self.off_road_since is not None:
            # Looking away has to be sustained before it is believed.
            # Looking back does too, and for the same reason: one
            # frame is not evidence of anything.
            #
            # A head does not travel fifty degrees and return inside a
            # tenth of a second, so a reading that says it did is a
            # bad landmark fit, not the driver. One such frame used to
            # end the episode outright -- on a recorded nine second
            # turn, a single frame reading 9 degrees between two
            # reading 54 threw away everything after it, and the turn
            # was never reported at all.
            if self.on_road_since is None:
                self.on_road_since = timestamp

            if timestamp - self.on_road_since >= self.recovery:
                self.off_road_since = None
                self.on_road_since = None

        off_road_duration = (
            0.0 if self.off_road_since is None
            else timestamp - self.off_road_since
        )

        distracted = off_road_duration >= self.min_duration

        if distracted and not self.distracted_active:
            self.distraction_count += 1

        self.distracted_active = distracted

        return {
            "eyes_on_road": not off_road,
            "off_road_duration": off_road_duration,
            "distracted": distracted,
            "distraction_count": self.distraction_count,
            "attention_yaw": yaw
        }

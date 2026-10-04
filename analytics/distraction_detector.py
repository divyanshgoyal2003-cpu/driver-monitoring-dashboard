from app.config import OFF_ROAD_YAW, OFF_ROAD_PITCH, DISTRACTION_DURATION


class DistractionDetector:
    """
    Flags the driver as distracted when their gaze stays off the road for
    DISTRACTION_DURATION seconds.

    The horizontal direction uses the combined head + eye yaw from
    GazeEstimator when available (head yaw alone otherwise); the vertical
    direction uses head pitch.
    """

    def __init__(self, max_yaw=OFF_ROAD_YAW, max_pitch=OFF_ROAD_PITCH,
                 min_duration=DISTRACTION_DURATION):

        self.max_yaw = max_yaw
        self.max_pitch = max_pitch
        self.min_duration = min_duration

        self.off_road_since = None
        self.distracted_active = False
        self.distraction_count = 0

    def update(self, head_pose, gaze, timestamp):

        if head_pose is None or not head_pose["calibrated"]:
            # No face, or the neutral pose is still being learned (any
            # offset now is the camera's, not the driver's).
            self.off_road_since = None
            self.distracted_active = False
            return None

        yaw = gaze["gaze_yaw"] if gaze else head_pose["yaw"]
        pitch = head_pose["pitch"]

        off_road = abs(yaw) > self.max_yaw or abs(pitch) > self.max_pitch

        if off_road:
            if self.off_road_since is None:
                self.off_road_since = timestamp

            off_road_duration = timestamp - self.off_road_since

        else:
            self.off_road_since = None
            off_road_duration = 0.0

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

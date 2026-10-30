from datetime import datetime

from app.config import METRICS_LOG_INTERVAL, DATABASE_COMMIT_INTERVAL


def _now():
    return datetime.now().isoformat(timespec="seconds")


def _value(state, section, key):
    """state[section][key], or None when that monitor has no data."""

    data = state.get(section)

    if not data:
        return None

    value = data.get(key)

    # sqlite3 cannot store NumPy integers; convert NumPy scalars to
    # plain Python values.
    if hasattr(value, "item"):
        value = value.item()

    return int(value) if isinstance(value, bool) else value


class SessionLogger:
    """
    Writes one monitoring session to the database: a metrics snapshot
    every METRICS_LOG_INTERVAL seconds, every event, and a summary when
    the session closes.
    """

    def __init__(self, database, source,
                 log_interval=METRICS_LOG_INTERVAL,
                 commit_interval=DATABASE_COMMIT_INTERVAL):

        self.database = database
        self.log_interval = log_interval
        self.commit_interval = commit_interval

        self.session_id = database.start_session(_now(), str(source))

        self.start_timestamp = None
        self.next_log = None
        self.last_commit = None
        self.elapsed = 0.0

        self.score_total = 0.0
        self.score_count = 0
        self.score_minimum = None

        # Latest cumulative counts reported by the monitors.
        self.counts = {
            "blinks": 0,
            "yawns": 0,
            "microsleeps": 0,
            "distractions": 0,
            "phone_uses": 0,
        }

    def log(self, state):

        timestamp = state["timestamp"]

        if self.start_timestamp is None:
            self.start_timestamp = timestamp

        self.elapsed = timestamp - self.start_timestamp

        recorded_at = _now()

        for event in state["risk"]["events"]:
            self.database.add_event(
                self.session_id,
                recorded_at,
                self.elapsed,
                event["type"],
                event["severity"],
                event["message"]
            )

        self._update_counts(state)

        # Snapshots follow a fixed grid (start, start + interval, ...), so
        # frame jitter does not make the spacing drift. The tolerance
        # absorbs floating-point error in the timestamps.
        if self.next_log is None or timestamp >= self.next_log - 1e-6:
            self.database.add_metrics(
                self.session_id,
                self._metrics(state, recorded_at)
            )
            self._track_score(state["risk"]["score"])
            self._schedule_next_log(timestamp)

        if (self.last_commit is None
                or timestamp - self.last_commit >= self.commit_interval):
            self.database.commit()
            self.last_commit = timestamp

    def _schedule_next_log(self, timestamp):

        if self.next_log is None:
            self.next_log = timestamp + self.log_interval
            return

        self.next_log += self.log_interval

        # After a stall, restart the grid instead of catching up with a
        # burst of snapshots.
        if self.next_log <= timestamp:
            self.next_log = timestamp + self.log_interval

    def close(self):

        average = (
            self.score_total / self.score_count
            if self.score_count
            else None
        )

        summary = {
            "ended_at": _now(),
            "duration": self.elapsed,
            "average_score": average,
            "minimum_score": self.score_minimum,
        }
        summary.update(self.counts)

        self.database.end_session(self.session_id, summary)

    def _metrics(self, state, recorded_at):

        risk = state["risk"]
        context = risk.get("context")

        # Head angles before calibration finishes are relative to the
        # camera, not to the driver's neutral pose; leave them out
        # rather than log a jump when calibration lands.
        head_pose = state.get("head_pose")

        if head_pose and not head_pose.get("calibrated"):
            state = dict(state, head_pose=None)

        return {
            "recorded_at": recorded_at,
            "elapsed": self.elapsed,
            "fps": state.get("fps"),
            "face_detected": int(bool(state.get("face_detected"))),
            "ear": _value(state, "blink", "ear"),
            "blink_rate": _value(state, "blink", "blink_rate"),
            "perclos": _value(state, "drowsiness", "perclos"),
            "drowsiness": _value(state, "drowsiness", "state"),
            "mar": _value(state, "yawn", "mar"),
            "yaw": _value(state, "head_pose", "yaw"),
            "pitch": _value(state, "head_pose", "pitch"),
            "roll": _value(state, "head_pose", "roll"),
            "head_direction": _value(state, "head_pose", "direction"),
            "gaze_direction": _value(state, "gaze", "direction"),
            "eyes_on_road": _value(state, "distraction", "eyes_on_road"),
            "phone_in_use": _value(state, "phone", "in_use"),
            "seatbelt": _value(state, "seatbelt", "status"),
            "emotion": _value(state, "emotion", "emotion"),
            "emotion_confidence": _value(state, "emotion", "confidence"),
            "safety_score": risk["score"],
            "risk_level": risk["level"],
            "brightness": state.get("brightness"),
            "ear_threshold": _value(state, "blink", "threshold"),
            "weights_mode": (
                "dynamic" if risk.get("dynamic") else "static"
            ),
            # Logged so a session can be replayed against the other
            # scoring mode and the two compared.
            "context_fatigue": (
                context.fatigue_multiplier() if context else None
            ),
            "context_attention": (
                context.attention_multiplier() if context else None
            ),
        }

    def _update_counts(self, state):

        sources = {
            "blinks": ("blink", "blink_count"),
            "yawns": ("yawn", "yawn_count"),
            "microsleeps": ("drowsiness", "microsleep_count"),
            "distractions": ("distraction", "distraction_count"),
            "phone_uses": ("phone", "usage_count"),
        }

        for name, (section, key) in sources.items():
            value = _value(state, section, key)

            if value is not None:
                self.counts[name] = value

    def _track_score(self, score):

        self.score_total += score
        self.score_count += 1

        if self.score_minimum is None or score < self.score_minimum:
            self.score_minimum = score

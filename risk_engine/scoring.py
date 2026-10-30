"""
Driver Safety Index.

Combines the output of every monitor into one score from 0 to 100
(100 = no risk detected), a risk level, the alerts to show and the
events to log.

A condition's cost is not a constant. Its entry in RISK_PENALTIES is
what it costs at its worst, and three layers scale that per frame:

    penalty = base * severity * confidence * context

Severity (how bad this instance is) and confidence (how far the
detector can be trusted) live in risk_engine.weights and can only
reduce a penalty. Context (time of day, time at the wheel, how much
light there is) lives in risk_engine.context and can only raise one. So
`base` stays the documented maximum and the 100-point budget keeps its
meaning.

Passing dynamic=False scores with the flat table instead, which is what
--static-weights runs. Both modes are kept so one recording can be put
through each and compared.
"""

import math

from risk_engine import weights
from app.config import (
    RISK_PENALTIES,
    SAFE_SCORE,
    CAUTION_SCORE,
    SCORE_FALL_TIME,
    SCORE_RISE_TIME,
    DRIVER_ABSENT_TIME,
    YAWN_ALERT_COUNT,
    DYNAMIC_WEIGHTS,
    FATIGUE_CONDITIONS,
    ATTENTION_CONDITIONS
)


class DriverSafetyIndex:

    # (condition, alert message, severity), most urgent first. The first
    # active condition is the one shown in the alert banner.
    CONDITIONS = (
        ("microsleep", "MICROSLEEP - WAKE UP", "critical"),
        ("phone", "PHONE IN USE", "high"),
        ("distracted", "EYES OFF ROAD", "high"),
        ("camera_blocked", "CAMERA BLOCKED", "high"),
        ("driver_absent", "DRIVER NOT VISIBLE", "high"),
        ("drowsy", "DROWSINESS DETECTED", "high"),
        ("tired", "SIGNS OF FATIGUE", "medium"),
        ("no_seatbelt", "SEATBELT NOT DETECTED", "medium"),
        ("frequent_yawning", "FREQUENT YAWNING", "medium"),
        ("negative_emotion", "NEGATIVE EMOTION", "low"),
    )

    def __init__(self, penalties=RISK_PENALTIES, safe_score=SAFE_SCORE,
                 caution_score=CAUTION_SCORE,
                 fall_time=SCORE_FALL_TIME, rise_time=SCORE_RISE_TIME,
                 absent_time=DRIVER_ABSENT_TIME,
                 yawn_alert_count=YAWN_ALERT_COUNT,
                 dynamic=DYNAMIC_WEIGHTS):

        self.dynamic = dynamic
        self.penalties = dict(penalties)
        self.safe_score = safe_score
        self.caution_score = caution_score
        self.fall_time = fall_time
        self.rise_time = rise_time
        self.absent_time = absent_time
        self.yawn_alert_count = yawn_alert_count

        self.score = None
        self.level = None
        self.last_timestamp = None
        self.active = set()
        self.face_lost_since = None
        self.last_yawn_count = 0

    def update(self, state, timestamp, context=None):
        """
        state:   VideoProcessor's per-frame state (face_detected,
                 drowsiness, yawn, distraction, phone, seatbelt,
                 emotion).
        context: a DrivingContext, or None to weigh every situation
                 alike.
        """

        active = self._active_conditions(state, timestamp)

        penalties, multipliers = self._weigh(
            active, state, context, self._absent_duration(timestamp)
        )

        raw_score = max(0.0, 100.0 - sum(penalties.values()))

        self._smooth(raw_score, timestamp)

        level = self._level(self.score)

        events = []

        for name, message, severity in self.CONDITIONS:
            if name in active and name not in self.active:
                events.append({
                    "type": name.upper(),
                    "severity": severity,
                    "message": message
                })

        yawn = state.get("yawn")

        if yawn and yawn["yawn_count"] > self.last_yawn_count:
            events.append({
                "type": "YAWN",
                "severity": "low",
                "message": f"Yawn #{yawn['yawn_count']}"
            })
            self.last_yawn_count = yawn["yawn_count"]

        if level == "DANGER" and self.level != "DANGER":
            events.append({
                "type": "HIGH_RISK",
                "severity": "high",
                "message": f"Safety index fell to {self.score:.0f}"
            })

        self.active = active
        self.level = level

        # The condition's name travels with the alert so a listener can
        # tell a drowsy driver from a distracted one. They share a
        # severity but not a remedy: one needs informing, the other
        # needs rousing.
        alerts = [
            {"name": name, "message": message, "severity": severity}
            for name, message, severity in self.CONDITIONS
            if name in active
        ]

        return {
            "score": self.score,
            "raw_score": raw_score,
            "level": level,
            "alerts": alerts,
            "events": events,
            "penalties": penalties,
            "multipliers": multipliers,
            "dynamic": self.dynamic,
            "context": context
        }

    def _weigh(self, active, state, context, absent_duration):
        """
        What each active condition costs this frame, and the three
        factors behind it. The factors are returned so the live panel
        can show why the score moved rather than only that it did.
        """

        penalties = {}
        multipliers = {}

        if not self.dynamic:
            return {name: self.penalties[name] for name in active}, {}

        darkness = context.darkness if context is not None else 0.0

        for name in active:

            severity = weights.severity(name, state, absent_duration)
            confidence = weights.confidence(name, state, darkness)
            amplifier = self._context_multiplier(name, context)

            penalties[name] = (
                self.penalties[name] * severity * confidence * amplifier
            )
            multipliers[name] = {
                "severity": severity,
                "confidence": confidence,
                "context": amplifier
            }

        return penalties, multipliers

    @staticmethod
    def _context_multiplier(name, context):
        """
        Fatigue conditions answer to the clock and to time at the
        wheel; attention conditions answer to how much the driver can
        see. Nothing else is amplified.
        """

        if context is None:
            return 1.0

        if name in FATIGUE_CONDITIONS:
            return context.fatigue_multiplier()

        if name in ATTENTION_CONDITIONS:
            return context.attention_multiplier()

        return 1.0

    def _absent_duration(self, timestamp):

        if self.face_lost_since is None:
            return 0.0

        return max(timestamp - self.face_lost_since, 0.0)

    def _active_conditions(self, state, timestamp):

        active = set()

        camera = state.get("camera")
        blocked = bool(camera and camera["blocked"])

        if blocked:
            active.add("camera_blocked")

        if state.get("face_detected"):
            self.face_lost_since = None

        else:
            if self.face_lost_since is None:
                self.face_lost_since = timestamp

            # A covered lens already explains the missing face. Raising
            # DRIVER NOT VISIBLE as well would charge one cause twice
            # and send the driver looking for a problem in the seat.
            if (timestamp - self.face_lost_since >= self.absent_time
                    and not blocked):
                active.add("driver_absent")

        drowsiness = state.get("drowsiness")

        if drowsiness:
            if drowsiness["state"] == "MICROSLEEP":
                active.add("microsleep")
            elif drowsiness["state"] == "DROWSY":
                active.add("drowsy")
            elif drowsiness["state"] == "TIRED":
                active.add("tired")

        yawn = state.get("yawn")

        if yawn and yawn["recent_yawns"] >= self.yawn_alert_count:
            active.add("frequent_yawning")

        distraction = state.get("distraction")

        if distraction and distraction["distracted"]:
            active.add("distracted")

        phone = state.get("phone")

        if phone and phone["in_use"]:
            active.add("phone")

        seatbelt = state.get("seatbelt")

        if seatbelt and seatbelt["status"] == "OFF":
            active.add("no_seatbelt")

        emotion = state.get("emotion")

        if emotion and emotion["negative"]:
            active.add("negative_emotion")

        return active

    def _smooth(self, raw_score, timestamp):

        if self.score is None or self.last_timestamp is None:
            self.score = raw_score

        else:
            dt = max(timestamp - self.last_timestamp, 0.0)
            time_constant = (
                self.fall_time if raw_score < self.score else self.rise_time
            )
            alpha = 1.0 - math.exp(-dt / time_constant)
            self.score += alpha * (raw_score - self.score)

        self.last_timestamp = timestamp

    def _level(self, score):

        if score >= self.safe_score:
            return "SAFE"

        if score >= self.caution_score:
            return "CAUTION"

        return "DANGER"

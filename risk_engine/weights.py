"""
Severity and confidence weights for the Driver Safety Index.

Both answer a question the flat penalty table cannot:

    severity   - how bad is this instance? A six second glance away is
                 worse than a two second one, and the old table charged
                 the same for both.
    confidence - how much is this detection worth? A seatbelt verdict
                 sitting on the boundary of a heuristic should not cost
                 as much as a measured one.

Both only ever return a value at or below 1.0, so a condition can never
cost more than its entry in RISK_PENALTIES. Amplification is the
context layer's job alone (see risk_engine.context).

Every function falls back to 1.0 when the measurement it needs is
absent, so a detector returning partial data weakens nothing.
"""

from app.config import (
    SEVERITY_FLOOR,
    SEVERITY_MICROSLEEP,
    SEVERITY_DROWSY,
    SEVERITY_TIRED,
    SEVERITY_DISTRACTED,
    SEVERITY_PHONE,
    SEVERITY_YAWNING,
    SEVERITY_ABSENT,
    CONFIDENCE_FLOOR,
    CONFIDENCE_SEATBELT_AT_BOUNDARY,
    CONFIDENCE_EMOTION,
    CONFIDENCE_PHONE,
    CONFIDENCE_EYES_DARK,
    EYE_CONDITIONS,
    SEATBELT_OFF_RATIO
)


def ramp(value, low, high):
    """0 at or below `low`, 1 at or above `high`, linear between."""

    if high == low:
        return 0.0 if value < high else 1.0

    return min(max((value - low) / (high - low), 0.0), 1.0)


def _scaled(value, span, floor):
    """`floor` at the bottom of `span`, 1.0 at the top."""

    low, high = span[0], span[1]

    return floor + (1.0 - floor) * ramp(value, low, high)


def _get(state, section, key):
    """state[section][key], or None when that monitor has no data."""

    data = state.get(section)

    if not data:
        return None

    return data.get(key)


# ----------------------------------------------------------------------
# severity
# ----------------------------------------------------------------------


def severity(condition, state, absent_duration=None):
    """
    How much of its base penalty this instance of `condition` deserves,
    between SEVERITY_FLOOR and 1.0.
    """

    if condition == "microsleep":
        # Measured by the blink detector, which owns the eye timing.
        return _from(_get(state, "blink", "closed_duration"),
                     SEVERITY_MICROSLEEP)

    if condition == "drowsy":
        return _from(_get(state, "drowsiness", "perclos"), SEVERITY_DROWSY)

    if condition == "tired":
        return _from(_get(state, "drowsiness", "perclos"), SEVERITY_TIRED)

    if condition == "distracted":
        return _from(_get(state, "distraction", "off_road_duration"),
                     SEVERITY_DISTRACTED)

    if condition == "phone":
        return _from(_get(state, "phone", "usage_duration"), SEVERITY_PHONE)

    if condition == "frequent_yawning":
        return _from(_get(state, "yawn", "recent_yawns"), SEVERITY_YAWNING)

    if condition == "driver_absent":
        return _from(absent_duration, SEVERITY_ABSENT)

    # no_seatbelt and negative_emotion are states rather than episodes:
    # there is no duration or magnitude that makes them worse.
    return 1.0


def _from(value, span):

    if value is None:
        return 1.0

    return _scaled(value, span, SEVERITY_FLOOR)


# ----------------------------------------------------------------------
# confidence
# ----------------------------------------------------------------------


def confidence(condition, state, darkness=0.0):
    """
    How much the detector behind `condition` can be trusted this frame,
    between CONFIDENCE_FLOOR and 1.0.
    """

    if condition in EYE_CONDITIONS:
        return _clamp(_eye_confidence(state, darkness))

    if condition == "no_seatbelt":
        return _clamp(_seatbelt_confidence(state))

    if condition == "negative_emotion":
        value = _get(state, "emotion", "confidence")

        if value is None:
            return 1.0

        return _clamp(_scaled(value, CONFIDENCE_EMOTION,
                              CONFIDENCE_EMOTION[2]))

    if condition == "phone":
        return _clamp(_phone_confidence(state))

    return 1.0


def _eye_confidence(state, darkness):
    """
    Eye measurements depend on readable eyelids. Two things erode them:
    a calibration the EAR calibrator could not complete well, and a dark
    cabin, where the landmarks get noisy.

    This is the counterweight to the circadian multiplier. Night raises
    the risk a drowsy driver poses, but it lowers the camera's ability
    to see that driver, and without this the engine would amplify its
    own noise at exactly the hour false alarms do the most damage.
    """

    calibration = _get(state, "blink", "eye_confidence")
    weight = 1.0 if calibration is None else calibration

    return weight * (1.0 - (1.0 - CONFIDENCE_EYES_DARK) * darkness)


def _seatbelt_confidence(state):
    """
    The pooled ratio is how many of the recent visible frames contained
    a strap. A verdict of OFF is most doubtful just under the boundary
    and most certain when nothing strap-shaped was seen at all.
    """

    ratio = _get(state, "seatbelt", "ratio")

    if ratio is None:
        return 1.0

    certainty = 1.0 - ramp(ratio, 0.0, SEATBELT_OFF_RATIO)

    return CONFIDENCE_SEATBELT_AT_BOUNDARY + (
        1.0 - CONFIDENCE_SEATBELT_AT_BOUNDARY
    ) * certainty


def _phone_confidence(state):

    phones = _get(state, "phone", "phones")

    if not phones:
        return 1.0

    best = max(p.get("confidence", 1.0) for p in phones)

    return _scaled(best, CONFIDENCE_PHONE, CONFIDENCE_PHONE[2])


def _clamp(value):

    return min(max(value, CONFIDENCE_FLOOR), 1.0)

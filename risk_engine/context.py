"""
Driving context: the part of the risk picture that is not the driver.

The same microsleep is not equally dangerous at every moment. Time of
day, time already spent at the wheel and how much the camera can see all
change how much weight a detection deserves. This module gathers what
the current sensor suite can actually observe and turns it into
multipliers for the risk engine.

Only multipliers of 1.0 or more are produced: context amplifies risk and
never discounts it, so the score can never be more permissive than the
flat baseline.

What is live here, and what is not, is deliberate. `hour`, `elapsed` and
`brightness` are measured. `weather` and `speed` are part of the
interface but can only be set by hand, because an in-cabin camera cannot
see the road and there is no vehicle bus to read: weather would need an
API and a location, speed an OBD-II link. They are carried so the
scoring already accepts them when a sensor appears.
"""

from dataclasses import dataclass
from datetime import datetime

from app.config import (
    BRIGHTNESS_DARK,
    BRIGHTNESS_LIT,
    CIRCADIAN_MULTIPLIERS,
    TIME_ON_TASK_HOURS,
    TIME_ON_TASK_MAX,
    LOW_LIGHT_ATTENTION_MAX
)


def _ramp(value, low, high):
    """0 at or below `low`, 1 at or above `high`, linear between."""

    if high == low:
        return 0.0 if value < high else 1.0

    return min(max((value - low) / (high - low), 0.0), 1.0)


@dataclass(frozen=True)
class DrivingContext:
    """
    hour       : 0-23, local time
    elapsed    : seconds since this session started
    brightness : mean frame luminance 0-255, or None when not measured
    weather    : manual override; "clear" unless set
    speed      : km/h, None without a vehicle link
    """

    hour: int = 12
    elapsed: float = 0.0
    brightness: float = None
    weather: str = "clear"
    speed: float = None

    @property
    def darkness(self):
        """0 in a well-lit cabin, 1 in a dark one."""

        if self.brightness is None:
            return 0.0

        return 1.0 - _ramp(self.brightness, BRIGHTNESS_DARK, BRIGHTNESS_LIT)

    def circadian_multiplier(self):
        """How much the time of day raises fatigue risk."""

        for start, end, multiplier in CIRCADIAN_MULTIPLIERS:
            if start <= self.hour < end:
                return multiplier

        return 1.0

    def time_on_task_multiplier(self):
        """How much time already at the wheel raises fatigue risk."""

        hours = self.elapsed / 3600.0

        return 1.0 + TIME_ON_TASK_MAX * _ramp(hours, 0.0, TIME_ON_TASK_HOURS)

    def fatigue_multiplier(self):

        return self.circadian_multiplier() * self.time_on_task_multiplier()

    def attention_multiplier(self):
        """
        Less visibility leaves less margin for a glance away, so the
        attention conditions weigh more in the dark.
        """

        return 1.0 + LOW_LIGHT_ATTENTION_MAX * self.darkness

    def summary(self):
        """Short label for the live panel, or None when nothing applies."""

        parts = []

        if self.circadian_multiplier() > 1.0:
            parts.append("night" if self.hour < 7 else "afternoon")

        if self.time_on_task_multiplier() > 1.01:
            parts.append(f"{self.elapsed / 3600.0:.1f}h")

        if self.darkness > 0.05:
            parts.append("dark")

        return " ".join(parts) if parts else None


class ContextProvider:
    """
    Builds a DrivingContext each frame. Tracks the session's start so
    `elapsed` is measured from the first frame rather than the clock.
    """

    def __init__(self, weather="clear", speed=None, hour=None, clock=None):
        """
        weather / speed / hour override what cannot be sensed (or, for
        hour, let a demo pretend it is 3am). clock is injectable for
        tests.
        """

        self.weather = weather
        self.speed = speed
        self.hour_override = hour
        self.clock = clock or datetime.now

        self.start_timestamp = None

    def update(self, timestamp, brightness=None):

        if self.start_timestamp is None:
            self.start_timestamp = timestamp

        hour = (
            self.hour_override
            if self.hour_override is not None
            else self.clock().hour
        )

        return DrivingContext(
            hour=hour,
            elapsed=max(timestamp - self.start_timestamp, 0.0),
            brightness=brightness,
            weather=self.weather,
            speed=self.speed
        )

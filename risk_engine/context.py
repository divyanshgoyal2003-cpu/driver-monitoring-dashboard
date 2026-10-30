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

What is measured here, and what is not, is deliberate. `hour`, `elapsed`
and `brightness` come from the clock and the frame. `weather` weighs the
score but its value can only be set by hand, because an in-cabin camera
cannot see the road: reading it would need a location and a forecast
API. `speed` is carried and not yet weighed at all - it would need an
OBD-II link, and without one there is nothing to calibrate a speed
multiplier against.
"""

from dataclasses import dataclass
from datetime import datetime

from app.config import (
    BRIGHTNESS_DARK,
    BRIGHTNESS_LIT,
    CIRCADIAN_MULTIPLIERS,
    TIME_ON_TASK_HOURS,
    TIME_ON_TASK_MAX,
    LOW_LIGHT_ATTENTION_MAX,
    WEATHER_MULTIPLIERS
)


def _ramp(value, low, high):
    """0 at or below `low`, 1 at or above `high`, linear between."""

    if high == low:
        return 0.0 if value < high else 1.0

    return min(max((value - low) / (high - low), 0.0), 1.0)


@dataclass(frozen=True)
class DrivingContext:
    """
    hour           : 0-23, local time
    elapsed        : seconds since this session started
    brightness     : mean frame luminance 0-255, or None when measured
    weather        : the condition in force; "clear" unless set
    weather_source : where that condition came from - "driver" when
                     declared, "forecast" when fetched, "default" when
                     nobody said. A multiplier is only as good as its
                     input, so the input's provenance travels with it
                     to the panel and into the session log.
    speed          : km/h, None without a vehicle link
    """

    hour: int = 12
    elapsed: float = 0.0
    brightness: float = None
    weather: str = "clear"
    weather_source: str = "default"
    speed: float = None

    @property
    def conditions(self):
        """The weather as a lookup key: lowercased and underscored."""

        key = "_".join(str(self.weather or "").lower().split())

        return key or "clear"

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

    def weather_multiplier(self):
        """
        How much the conditions outside raise the cost of a lapse.

        Unlike the circadian and time-on-task multipliers, this says
        nothing about how likely a lapse is - rain does not make a
        driver sleepier. It is about consequence: there is less room to
        recover on a wet road, so the same lapse is worth more. That
        argument holds for a microsleep and a glance away alike, so
        this amplifies both families.
        """

        return WEATHER_MULTIPLIERS.get(self.conditions, 1.0)

    def fatigue_multiplier(self):

        return (
            self.circadian_multiplier()
            * self.time_on_task_multiplier()
            * self.weather_multiplier()
        )

    def attention_multiplier(self):
        """
        Less visibility leaves less margin for a glance away, so the
        attention conditions weigh more in the dark and in bad weather.
        """

        return (
            (1.0 + LOW_LIGHT_ATTENTION_MAX * self.darkness)
            * self.weather_multiplier()
        )

    def summary(self):
        """Short label for the live panel, or None when nothing applies."""

        parts = []

        if self.circadian_multiplier() > 1.0:
            parts.append("night" if self.hour < 7 else "afternoon")

        if self.time_on_task_multiplier() > 1.01:
            parts.append(f"{self.elapsed / 3600.0:.1f}h")

        if self.weather_multiplier() > 1.0:
            parts.append(self.conditions.replace("_", " "))

        if self.darkness > 0.05:
            parts.append("dark")

        return " ".join(parts) if parts else None


class ContextProvider:
    """
    Builds a DrivingContext each frame. Tracks the session's start so
    `elapsed` is measured from the first frame rather than the clock.
    """

    # What cycle_weather steps through: the forecast first, then each
    # condition in turn.
    WEATHER_CYCLE = (None,) + tuple(WEATHER_MULTIPLIERS)

    def __init__(self, weather=None, speed=None, hour=None, clock=None,
                 weather_service=None):
        """
        weather is the driver's own declaration, and None means they
        have not made one - which is what lets the forecast speak.
        speed and hour override what cannot be sensed (or, for hour,
        let a demo pretend it is 3am). weather_service and clock are
        injectable for tests.
        """

        self.declared = weather
        self.weather_service = weather_service
        self.speed = speed
        self.hour_override = hour
        self.clock = clock or datetime.now

        self.start_timestamp = None

    def override(self, weather):
        """
        The driver disagrees with the forecast. None hands control back
        to it.
        """

        self.declared = weather

    def cycle_weather(self):
        """
        Step the override on by one, for the live window's W key, and
        return what is now in force. None means the forecast again.
        """

        current = (
            self.WEATHER_CYCLE.index(self.declared)
            if self.declared in self.WEATHER_CYCLE
            else 0
        )

        self.declared = self.WEATHER_CYCLE[
            (current + 1) % len(self.WEATHER_CYCLE)
        ]

        return self.declared

    def _weather(self):
        """
        (condition, source). The driver's word beats the forecast, and
        the forecast beats nothing.

        A forecast that has not arrived or has stopped arriving falls
        back to "clear", which is the neutral weight: a system that
        cannot see the weather must not invent an amplification. The
        source says so, so a 1.0 from ignorance is never mistaken on
        the panel for a 1.0 from a clear sky.
        """

        if self.declared is not None:
            return self.declared, "driver"

        if self.weather_service is not None:
            observed, label = self.weather_service.provenance()

            return (observed if observed is not None else "clear"), label

        return "clear", "default"

    def update(self, timestamp, brightness=None):

        if self.start_timestamp is None:
            self.start_timestamp = timestamp

        hour = (
            self.hour_override
            if self.hour_override is not None
            else self.clock().hour
        )

        weather, source = self._weather()

        return DrivingContext(
            hour=hour,
            elapsed=max(timestamp - self.start_timestamp, 0.0),
            brightness=brightness,
            weather=weather,
            weather_source=source,
            speed=self.speed
        )

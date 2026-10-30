"""
Asking a forecast what the weather is, without trusting it too far.

The weather multiplier needs a value, and until now the only source was
the driver typing one. A forecast is a better default than nothing, but
it is worth being exact about what it is and is not:

  - It answers for a POINT, at city resolution. Rain five kilometres
    away reads as rain here.
  - It is up to fifteen minutes old, which is the API's own refresh.
  - It says nothing about the road surface, which is the thing that
    actually matters. Standing water and black ice are invisible to it.

So the forecast is a default, not an authority. The driver can overrule
it from the live window, and every reading carries where it came from so
a declared condition is never mistaken in the record for a measured one.

Nothing here is allowed to disturb the frame loop. The fetch runs on a
daemon thread and the loop only ever reads a cached value, so a slow or
dead network costs no frames. A failure keeps the last good reading and
says so rather than quietly reporting clear skies.
"""

import json
import sys
import threading
import urllib.parse
import urllib.request

from app.config import (
    WEATHER_URL,
    WEATHER_REFRESH,
    WEATHER_TIMEOUT,
    WEATHER_MULTIPLIERS,
    WMO_WEATHER,
    ICE_TEMPERATURE,
    FOG_VISIBILITY
)

# What the service is doing, for the panel.
OFF = "off"
WAITING = "waiting"
OK = "ok"
STALE = "stale"
FAILED = "failed"


def _worse(first, second):
    """Whichever of two conditions leaves less margin for a lapse."""

    if WEATHER_MULTIPLIERS.get(second, 1.0) > WEATHER_MULTIPLIERS.get(first, 1.0):
        return second

    return first


def classify(observation):
    """
    One of the seven conditions in WEATHER_MULTIPLIERS, or None when the
    forecast cannot be interpreted.

    None matters: an unrecognised code must not become "clear", because
    that would silently drop the amplification instead of admitting the
    reading was not understood.
    """

    if not observation:
        return None

    weather = WMO_WEATHER.get(observation.get("weather_code"))

    if weather is None:
        return None

    temperature = observation.get("temperature_2m")
    visibility = observation.get("visibility")

    # Rain at freezing is ice on the road whatever the forecast calls
    # it: liquid water landing on sub-zero tarmac is the black-ice
    # case, and a code of plain "rain" at -1C hides it.
    #
    # Snow is deliberately NOT included. It is already frozen when it
    # lands, and fresh snow grips better than the glaze freezing rain
    # leaves, so snow stays snow. Including it would also have made the
    # snow category unreachable, since snow falls at freezing by
    # definition.
    liquid = weather in ("rain", "heavy_rain")

    if liquid and temperature is not None and temperature <= ICE_TEMPERATURE:
        weather = _worse(weather, "ice")

    if visibility is not None and visibility <= FOG_VISIBILITY:
        weather = _worse(weather, "fog")

    return weather


def fetch_open_meteo(latitude, longitude, timeout=WEATHER_TIMEOUT,
                     url=WEATHER_URL):
    """The `current` block from Open-Meteo. Raises on any failure."""

    query = urllib.parse.urlencode({
        "latitude": latitude,
        "longitude": longitude,
        "current": "weather_code,temperature_2m,visibility"
    })

    with urllib.request.urlopen(f"{url}?{query}", timeout=timeout) as response:
        payload = json.load(response)

    return payload.get("current") or {}


class WeatherService:
    """
    A cached condition, refreshed on a daemon thread.

    `fetch` is injectable so the tests never touch the network.
    """

    def __init__(self, latitude=None, longitude=None,
                 refresh=WEATHER_REFRESH, timeout=WEATHER_TIMEOUT,
                 fetch=None):

        self.latitude = latitude
        self.longitude = longitude
        self.refresh = refresh
        self.timeout = timeout

        self._fetch = fetch or fetch_open_meteo

        self._lock = threading.Lock()
        self._weather = None
        self._status = OFF
        self._error = None

        # So an outage is reported once rather than every refresh.
        self._reported = False

        self._stop = threading.Event()
        self._thread = None

    @property
    def configured(self):
        return self.latitude is not None and self.longitude is not None

    def start(self):
        """
        Begin refreshing. False when no coordinates are set, which is
        not a failure: it is the default, and the driver's own setting
        then does all the work.
        """

        if not self.configured or self._thread is not None:
            return False

        with self._lock:
            self._status = WAITING

        self._thread = threading.Thread(
            target=self._run,
            name="weather",
            daemon=True
        )
        self._thread.start()

        return True

    def stop(self):
        self._stop.set()

    def reading(self):
        """
        (condition or None, status). Never blocks and never raises:
        this is called from the frame loop.
        """

        with self._lock:
            return self._weather, self._status

    def provenance(self):
        """
        (condition or None, a label saying how good the reading is).

        The label is what the panel and the session log show, so the
        vocabulary for describing a forecast stays in one place rather
        than being reinvented by every caller.
        """

        if not self.configured:
            # No coordinates were ever given, so there is no forecast to
            # be off or offline. Saying so would describe a feature the
            # driver never asked for.
            return None, "default"

        weather, status = self.reading()

        if weather is not None:
            return weather, "forecast" if status == OK else "forecast (stale)"

        return None, {
            OFF: "forecast (off)",
            WAITING: "forecast (waiting)",
        }.get(status, "forecast (offline)")

    @property
    def error(self):
        with self._lock:
            return self._error

    def _run(self):

        while not self._stop.is_set():
            self._refresh_once()

            # Event.wait rather than sleep, so stopping is immediate.
            self._stop.wait(self.refresh)

    def _refresh_once(self):

        try:
            weather = classify(
                self._fetch(self.latitude, self.longitude, self.timeout)
            )

        except Exception as error:
            # Deliberately broad. urllib, json and a changed response
            # shape all fail differently, and none of them is worth
            # taking the driver monitor down for. The last good reading
            # is kept and the status says it is no longer fresh.
            self._fail(f"{type(error).__name__}: {error}")

            return

        if weather is None:
            # Reached the API but could not read the answer.
            self._fail("unrecognised forecast code")

            return

        with self._lock:
            self._weather = weather
            self._status = OK
            self._error = None

            # A later outage is worth hearing about again.
            self._reported = False

    def _fail(self, reason):
        """
        Record an outage, and say so once.

        The panel can only show that the forecast is offline, not why.
        A wrong coordinate and an unplugged network look identical
        there, and both are things the driver can act on.
        """

        with self._lock:
            self._error = reason
            self._status = STALE if self._weather is not None else FAILED

            announce = not self._reported
            self._reported = True

        if announce:
            print(
                f"Weather forecast unavailable ({reason}). Press W to "
                f"declare the conditions yourself.",
                file=sys.stderr
            )

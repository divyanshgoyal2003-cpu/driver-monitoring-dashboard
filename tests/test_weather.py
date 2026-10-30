"""
The forecast, and the driver's right to overrule it.

Nothing here touches the network: WeatherService takes its fetch as an
argument, so these drive it with canned responses and deliberate
failures.
"""

import io
import threading
import unittest
from contextlib import redirect_stderr

from utils import weather as weather_module
from utils.weather import WeatherService, classify
from app.config import WMO_WEATHER, FOG_VISIBILITY


def observation(code, temperature=15.0, visibility=20000.0):
    return {
        "weather_code": code,
        "temperature_2m": temperature,
        "visibility": visibility
    }


class ClassifyTest(unittest.TestCase):

    def test_clear_sky(self):
        self.assertEqual(classify(observation(0)), "clear")

    def test_overcast_is_cloudy(self):
        self.assertEqual(classify(observation(3)), "cloudy")

    def test_moderate_rain(self):
        self.assertEqual(classify(observation(63)), "rain")

    def test_heavy_rain(self):
        self.assertEqual(
            classify(observation(65)), "heavy_rain"
        )

    def test_fog(self):
        self.assertEqual(classify(observation(45, visibility=200.0)), "fog")

    def test_snow(self):
        self.assertEqual(
            classify(observation(73, temperature=-2.0)),
            "snow"
        )

    def test_freezing_rain_is_ice(self):
        self.assertEqual(
            classify(observation(67, temperature=-1.0)),
            "ice"
        )

    def test_thunderstorms_count_as_heavy_rain(self):
        self.assertEqual(
            classify(observation(95)), "heavy_rain"
        )

    def test_every_mapped_code_gives_a_known_condition(self):
        from app.config import WEATHER_MULTIPLIERS

        for code in WMO_WEATHER:
            self.assertIn(classify(observation(code)), WEATHER_MULTIPLIERS)

    def test_an_unknown_code_is_no_reading_rather_than_clear(self):
        """
        The important one. Reporting "clear" for a code the table has
        never seen would silently drop the amplification and look
        exactly like a fine day.
        """

        self.assertIsNone(classify(observation(4242)))

    def test_a_missing_observation_is_no_reading(self):
        self.assertIsNone(classify({}))
        self.assertIsNone(classify(None))

    def test_a_missing_code_is_no_reading(self):
        self.assertIsNone(classify({"temperature_2m": 5.0}))


class RefinementTest(unittest.TestCase):
    """
    The two places the reported code understates the road.
    """

    def test_rain_at_freezing_is_upgraded_to_ice(self):
        warm = classify(observation(63, temperature=8.0))
        freezing = classify(observation(63, temperature=-1.0))

        self.assertEqual(warm, "rain")
        self.assertEqual(freezing, "ice")

    def test_cold_and_dry_is_not_ice(self):
        """
        A clear winter morning is not an ice warning. Something has to
        be falling.
        """

        self.assertEqual(
            classify(observation(0, temperature=-5.0)),
            "clear"
        )

    def test_snow_at_freezing_stays_snow(self):
        """
        Regression. The freezing upgrade first applied to anything
        falling, which fired on every snow reading - snow falls at
        freezing by definition - and made the snow category
        unreachable from a forecast. Snow lands already frozen and
        grips better than a freezing-rain glaze; it is not ice.
        """

        for temperature in (0.0, -2.0, -15.0):
            self.assertEqual(
                classify(observation(75, temperature=temperature)), "snow"
            )

    def test_low_visibility_is_upgraded_to_fog(self):
        clear_air = classify(observation(1, visibility=20000.0))
        thick = classify(observation(1, visibility=FOG_VISIBILITY / 2))

        self.assertEqual(clear_air, "clear")
        self.assertEqual(thick, "fog")

    def test_a_refinement_only_ever_picks_the_worse_reading(self):
        """
        Snow already weighs more than fog, so poor visibility during
        snow must not quietly downgrade it.
        """

        snow_in_fog = classify(observation(
            75, temperature=-3.0, visibility=100.0
        ))

        self.assertEqual(snow_in_fog, "snow")


class WeatherServiceTest(unittest.TestCase):

    def test_without_coordinates_it_does_nothing(self):
        service = WeatherService()

        self.assertFalse(service.configured)
        self.assertFalse(service.start())
        self.assertEqual(service.reading(), (None, weather_module.OFF))

    def test_a_good_fetch_is_reported(self):
        service = WeatherService(
            latitude=51.5,
            longitude=-0.1,
            fetch=lambda lat, lon, timeout: observation(63)
        )

        service._refresh_once()

        self.assertEqual(service.reading(), ("rain", weather_module.OK))

    def test_the_coordinates_are_passed_to_the_fetch(self):
        seen = {}

        def fetch(latitude, longitude, timeout):
            seen.update(latitude=latitude, longitude=longitude)
            return observation(0)

        WeatherService(latitude=12.5, longitude=-3.25, fetch=fetch)\
            ._refresh_once()

        self.assertEqual(seen, {"latitude": 12.5, "longitude": -3.25})

    def test_a_failed_fetch_does_not_raise(self):
        """
        This runs on a thread beside the frame loop. A network error
        must never become the monitor's problem.
        """

        def explode(latitude, longitude, timeout):
            raise OSError("no route to host")

        service = WeatherService(latitude=1.0, longitude=1.0, fetch=explode)

        with redirect_stderr(io.StringIO()):
            service._refresh_once()

        weather, status = service.reading()

        self.assertIsNone(weather)
        self.assertEqual(status, weather_module.FAILED)
        self.assertIn("no route to host", service.error)

    def test_a_failure_after_a_success_keeps_the_last_reading(self):
        responses = [observation(73, temperature=-2.0)]

        def fetch(latitude, longitude, timeout):
            if responses:
                return responses.pop()
            raise TimeoutError("timed out")

        service = WeatherService(latitude=1.0, longitude=1.0, fetch=fetch)

        service._refresh_once()
        self.assertEqual(service.reading(), ("snow", weather_module.OK))

        with redirect_stderr(io.StringIO()):
            service._refresh_once()

        self.assertEqual(service.reading(), ("snow", weather_module.STALE))

    def test_an_uninterpretable_answer_is_not_a_reading(self):
        service = WeatherService(
            latitude=1.0,
            longitude=1.0,
            fetch=lambda lat, lon, timeout: observation(4242)
        )

        with redirect_stderr(io.StringIO()):
            service._refresh_once()

        self.assertEqual(service.reading(), (None, weather_module.FAILED))

    def test_an_outage_is_announced_once_not_every_refresh(self):
        """
        It retries every fifteen minutes for the whole drive. Saying so
        each time would bury everything else on the console.
        """

        def explode(latitude, longitude, timeout):
            raise OSError("no route to host")

        service = WeatherService(latitude=1.0, longitude=1.0, fetch=explode)
        stderr = io.StringIO()

        with redirect_stderr(stderr):
            for _ in range(4):
                service._refresh_once()

        said = stderr.getvalue().strip().splitlines()

        self.assertEqual(len(said), 1)
        self.assertIn("no route to host", said[0])

    def test_a_later_outage_is_announced_again(self):
        responses = [observation(0)]

        def fetch(latitude, longitude, timeout):
            if responses:
                return responses.pop()
            raise OSError("gone again")

        service = WeatherService(latitude=1.0, longitude=1.0, fetch=fetch)
        stderr = io.StringIO()

        with redirect_stderr(stderr):
            service._refresh_once()   # works, clears the reported flag
            service._refresh_once()   # fails, worth hearing about

        self.assertIn("gone again", stderr.getvalue())

    def test_an_unconfigured_service_is_not_called_a_forecast(self):
        """
        Without coordinates there is no forecast to be off or offline.
        Reporting one would describe a feature the driver never asked
        for, and would keep the panel's WEATHER row up for no reason.
        """

        self.assertEqual(
            WeatherService().provenance(), (None, "default")
        )

    def test_provenance_labels_each_state(self):
        service = WeatherService(latitude=1.0, longitude=1.0)

        self.assertEqual(service.provenance(), (None, "forecast (off)"))

        service._status = weather_module.WAITING
        self.assertEqual(service.provenance(), (None, "forecast (waiting)"))

        service._status = weather_module.FAILED
        self.assertEqual(service.provenance(), (None, "forecast (offline)"))

        service._weather = "rain"
        service._status = weather_module.OK
        self.assertEqual(service.provenance(), ("rain", "forecast"))

        service._status = weather_module.STALE
        self.assertEqual(service.provenance(), ("rain", "forecast (stale)"))

    def test_the_thread_starts_and_stops(self):
        done = threading.Event()

        def fetch(latitude, longitude, timeout):
            done.set()
            return observation(0)

        service = WeatherService(
            latitude=1.0, longitude=1.0, refresh=0.01, fetch=fetch
        )

        self.assertTrue(service.start())
        self.assertTrue(done.wait(2.0))

        service.stop()
        service._thread.join(2.0)

        self.assertFalse(service._thread.is_alive())

    def test_starting_twice_does_not_make_a_second_thread(self):
        service = WeatherService(
            latitude=1.0,
            longitude=1.0,
            refresh=0.01,
            fetch=lambda lat, lon, timeout: observation(0)
        )

        service.start()
        first = service._thread

        self.assertFalse(service.start())
        self.assertIs(service._thread, first)

        service.stop()


if __name__ == "__main__":
    unittest.main()

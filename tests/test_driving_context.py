import unittest
from datetime import datetime

from risk_engine.context import DrivingContext, ContextProvider
from app.config import (
    BRIGHTNESS_DARK,
    BRIGHTNESS_LIT,
    TIME_ON_TASK_MAX,
    WEATHER_MULTIPLIERS
)


class DrivingContextTest(unittest.TestCase):

    def test_daytime_is_neutral(self):
        context = DrivingContext(hour=11)

        self.assertEqual(context.circadian_multiplier(), 1.0)
        self.assertEqual(context.fatigue_multiplier(), 1.0)
        self.assertEqual(context.attention_multiplier(), 1.0)

    def test_the_small_hours_weigh_fatigue_most(self):
        night = DrivingContext(hour=3).circadian_multiplier()
        shoulder = DrivingContext(hour=1).circadian_multiplier()
        afternoon = DrivingContext(hour=15).circadian_multiplier()
        midday = DrivingContext(hour=11).circadian_multiplier()

        self.assertGreater(night, shoulder)
        self.assertGreater(shoulder, afternoon)
        self.assertGreater(afternoon, midday)

    def test_every_hour_is_covered_and_never_discounts(self):
        for hour in range(24):
            self.assertGreaterEqual(
                DrivingContext(hour=hour).circadian_multiplier(), 1.0
            )

    def test_time_at_the_wheel_builds_up_then_levels_off(self):
        fresh = DrivingContext(elapsed=0).time_on_task_multiplier()
        hour = DrivingContext(elapsed=3600).time_on_task_multiplier()
        long = DrivingContext(elapsed=4 * 3600).time_on_task_multiplier()

        self.assertEqual(fresh, 1.0)
        self.assertGreater(hour, fresh)
        self.assertAlmostEqual(long, 1.0 + TIME_ON_TASK_MAX)

    def test_darkness_from_brightness(self):
        self.assertEqual(DrivingContext(brightness=BRIGHTNESS_LIT).darkness,
                         0.0)
        self.assertEqual(DrivingContext(brightness=BRIGHTNESS_DARK).darkness,
                         1.0)
        self.assertEqual(DrivingContext(brightness=5).darkness, 1.0)
        self.assertEqual(DrivingContext(brightness=250).darkness, 0.0)

    def test_unmeasured_brightness_is_not_treated_as_dark(self):
        self.assertEqual(DrivingContext(brightness=None).darkness, 0.0)
        self.assertEqual(
            DrivingContext(brightness=None).attention_multiplier(), 1.0
        )

    def test_a_dark_cabin_weighs_attention_more(self):
        dark = DrivingContext(brightness=10).attention_multiplier()
        lit = DrivingContext(brightness=200).attention_multiplier()

        self.assertGreater(dark, lit)

    def test_light_does_not_touch_fatigue_and_the_clock_does_not_touch_attention(self):
        dark_noon = DrivingContext(hour=12, brightness=10)
        lit_night = DrivingContext(hour=3, brightness=200)

        self.assertEqual(dark_noon.fatigue_multiplier(), 1.0)
        self.assertEqual(lit_night.attention_multiplier(), 1.0)

    def test_context_never_discounts(self):
        for hour in range(24):
            for brightness in (None, 0, 50, 255):
                for weather in tuple(WEATHER_MULTIPLIERS) + ("", None, "?"):
                    context = DrivingContext(
                        hour=hour,
                        elapsed=0,
                        brightness=brightness,
                        weather=weather
                    )
                    self.assertGreaterEqual(
                        context.fatigue_multiplier(), 1.0
                    )
                    self.assertGreaterEqual(
                        context.attention_multiplier(), 1.0
                    )

    def test_summary_names_what_is_raising_the_weight(self):
        self.assertIsNone(DrivingContext(hour=11, brightness=200).summary())
        self.assertIn("night", DrivingContext(hour=3).summary())
        self.assertIn("dark", DrivingContext(brightness=10).summary())


class WeatherTest(unittest.TestCase):
    """
    Weather weighs consequence rather than likelihood, so unlike the
    clock and the cabin light it amplifies *both* families: there is
    less room to recover from any lapse on a wet road.
    """

    def test_clear_weather_is_neutral(self):
        context = DrivingContext(hour=12, brightness=200, weather="clear")

        self.assertEqual(context.weather_multiplier(), 1.0)
        self.assertEqual(context.fatigue_multiplier(), 1.0)
        self.assertEqual(context.attention_multiplier(), 1.0)

    def test_rain_weighs_both_fatigue_and_attention(self):
        clear = DrivingContext(hour=12, brightness=200, weather="clear")
        rain = DrivingContext(hour=12, brightness=200, weather="rain")

        self.assertGreater(
            rain.fatigue_multiplier(), clear.fatigue_multiplier()
        )
        self.assertGreater(
            rain.attention_multiplier(), clear.attention_multiplier()
        )

    def test_worse_conditions_weigh_more(self):
        """
        Ordered by how much margin the road and the air leave, not by
        how unpleasant the drive is.
        """

        def weight(weather):
            return DrivingContext(weather=weather).weather_multiplier()

        self.assertEqual(weight("cloudy"), weight("clear"))
        self.assertGreater(weight("rain"), weight("clear"))
        self.assertGreater(weight("heavy_rain"), weight("rain"))
        self.assertGreater(weight("snow"), weight("heavy_rain"))
        self.assertGreater(weight("ice"), weight("snow"))

    def test_an_unknown_condition_weighs_nothing(self):
        """
        A forecast API will one day return words this table has never
        heard of. Guessing at them would inflate the score silently.
        """

        self.assertEqual(
            DrivingContext(weather="volcanic ash").weather_multiplier(), 1.0
        )

    def test_a_missing_condition_is_treated_as_clear(self):
        for weather in (None, "", "   "):
            context = DrivingContext(weather=weather)

            self.assertEqual(context.conditions, "clear")
            self.assertEqual(context.weather_multiplier(), 1.0)

    def test_case_and_spacing_do_not_matter(self):
        plain = DrivingContext(weather="heavy_rain").weather_multiplier()

        for spelling in ("Heavy Rain", "HEAVY_RAIN", "  heavy rain  "):
            self.assertEqual(
                DrivingContext(weather=spelling).weather_multiplier(),
                plain
            )

    def test_weather_compounds_with_the_clock(self):
        night_clear = DrivingContext(hour=3, weather="clear")
        night_snow = DrivingContext(hour=3, weather="snow")

        self.assertGreater(
            night_snow.fatigue_multiplier(),
            night_clear.fatigue_multiplier()
        )

    def test_weather_compounds_with_the_cabin_light(self):
        dark_clear = DrivingContext(brightness=10, weather="clear")
        dark_rain = DrivingContext(brightness=10, weather="rain")

        self.assertGreater(
            dark_rain.attention_multiplier(),
            dark_clear.attention_multiplier()
        )

    def test_summary_names_the_weather(self):
        summary = DrivingContext(hour=12, brightness=200,
                                 weather="snow").summary()

        self.assertEqual(summary, "snow")

    def test_summary_spells_the_weather_readably(self):
        summary = DrivingContext(weather="heavy_rain").summary()

        self.assertIn("heavy rain", summary)

    def test_summary_leaves_out_weather_that_changes_nothing(self):
        for weather in ("clear", "cloudy", "volcanic ash"):
            context = DrivingContext(hour=12, brightness=200,
                                     weather=weather)

            self.assertIsNone(context.summary())


class FakeForecast:
    """Stands in for a WeatherService; only provenance() is used."""

    def __init__(self, weather, label="forecast"):
        self.weather = weather
        self.label = label

    def provenance(self):
        return self.weather, self.label


class WeatherProvenanceTest(unittest.TestCase):
    """
    Where the condition came from, and who wins when two sources
    disagree. The driver can see out of the window; the forecast is
    answering for a point several kilometres away.
    """

    def provide(self, **kwargs):
        provider = ContextProvider(
            clock=lambda: datetime(2026, 1, 1, 12), **kwargs
        )

        return provider, provider.update(0.0)

    def test_nothing_declared_and_no_forecast_is_clear_by_default(self):
        _, context = self.provide()

        self.assertEqual(context.weather, "clear")
        self.assertEqual(context.weather_source, "default")

    def test_the_forecast_speaks_when_the_driver_does_not(self):
        _, context = self.provide(weather_service=FakeForecast("rain"))

        self.assertEqual(context.weather, "rain")
        self.assertEqual(context.weather_source, "forecast")
        self.assertGreater(context.weather_multiplier(), 1.0)

    def test_the_driver_overrules_the_forecast(self):
        _, context = self.provide(
            weather="snow", weather_service=FakeForecast("clear")
        )

        self.assertEqual(context.weather, "snow")
        self.assertEqual(context.weather_source, "driver")

    def test_the_driver_can_overrule_downwards_too(self):
        """
        The forecast says it is raining five kilometres away; here the
        road is dry. Saying so returns the baseline weight, and the
        record shows it was the driver's call.
        """

        _, context = self.provide(
            weather="clear", weather_service=FakeForecast("heavy_rain")
        )

        self.assertEqual(context.weather_multiplier(), 1.0)
        self.assertEqual(context.weather_source, "driver")

    def test_an_absent_forecast_is_neutral_but_says_so(self):
        """
        A 1.0 from ignorance must not be recorded as a 1.0 from a clear
        sky. The weight is the same; the provenance is not.
        """

        _, context = self.provide(
            weather_service=FakeForecast(None, "forecast (offline)")
        )

        self.assertEqual(context.weather, "clear")
        self.assertEqual(context.weather_multiplier(), 1.0)
        self.assertEqual(context.weather_source, "forecast (offline)")

    def test_overriding_and_handing_control_back(self):
        provider, _ = self.provide(weather_service=FakeForecast("rain"))

        provider.override("ice")
        self.assertEqual(provider.update(0.0).weather, "ice")

        provider.override(None)
        self.assertEqual(provider.update(0.0).weather, "rain")

    def test_the_cycle_starts_from_the_forecast_and_returns_to_it(self):
        provider, _ = self.provide(weather_service=FakeForecast("rain"))

        seen = []

        for _ in range(len(provider.WEATHER_CYCLE)):
            seen.append(provider.cycle_weather())

        self.assertEqual(seen[0], "clear")
        self.assertIsNone(seen[-1])
        self.assertEqual(provider.update(0.0).weather_source, "forecast")

    def test_the_cycle_covers_every_condition(self):
        provider, _ = self.provide()

        seen = {
            provider.cycle_weather()
            for _ in range(len(provider.WEATHER_CYCLE))
        }

        self.assertEqual(seen, set(WEATHER_MULTIPLIERS) | {None})

    def test_the_cycle_recovers_from_an_unknown_starting_value(self):
        provider, _ = self.provide(weather="volcanic ash")

        self.assertEqual(provider.cycle_weather(), "clear")


class ContextProviderTest(unittest.TestCase):

    def test_elapsed_runs_from_the_first_frame(self):
        provider = ContextProvider(clock=lambda: datetime(2026, 1, 1, 12))

        provider.update(1000.0)
        context = provider.update(1060.0)

        self.assertAlmostEqual(context.elapsed, 60.0)

    def test_hour_override_beats_the_clock(self):
        provider = ContextProvider(
            hour=3, clock=lambda: datetime(2026, 1, 1, 14)
        )

        self.assertEqual(provider.update(0.0).hour, 3)

    def test_the_clock_is_used_without_an_override(self):
        provider = ContextProvider(clock=lambda: datetime(2026, 1, 1, 14))

        self.assertEqual(provider.update(0.0).hour, 14)

    def test_unsensed_inputs_are_carried_through(self):
        provider = ContextProvider(weather="rain", speed=80.0)
        context = provider.update(0.0, brightness=120.0)

        self.assertEqual(context.weather, "rain")
        self.assertEqual(context.speed, 80.0)
        self.assertEqual(context.brightness, 120.0)


if __name__ == "__main__":
    unittest.main()

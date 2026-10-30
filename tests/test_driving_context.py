import unittest
from datetime import datetime

from risk_engine.context import DrivingContext, ContextProvider
from app.config import BRIGHTNESS_DARK, BRIGHTNESS_LIT, TIME_ON_TASK_MAX


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
                context = DrivingContext(
                    hour=hour, elapsed=0, brightness=brightness
                )
                self.assertGreaterEqual(context.fatigue_multiplier(), 1.0)
                self.assertGreaterEqual(context.attention_multiplier(), 1.0)

    def test_summary_names_what_is_raising_the_weight(self):
        self.assertIsNone(DrivingContext(hour=11, brightness=200).summary())
        self.assertIn("night", DrivingContext(hour=3).summary())
        self.assertIn("dark", DrivingContext(brightness=10).summary())


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

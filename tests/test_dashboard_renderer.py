"""
The panel's key hints, and the weather row.

The first test here exists because of a real miss: the W key for
overruling the forecast was wired into the live loop and written up in
the README, but never added to the hints along the bottom of the panel.
The driver had the control and no way to find it. Reading main.py for
the keys it handles is a little unusual for a unit test, but it is the
only thing that actually keeps the two in step.
"""

import pathlib
import re
import unittest

from app.config import PANEL_WIDTH
from renderers.dashboard_renderer import (
    DashboardRenderer,
    KEY_HINTS,
    LABEL_X
)
from renderers.theme import text_width
from risk_engine.context import DrivingContext

MAIN = pathlib.Path(__file__).resolve().parent.parent / "app" / "main.py"


def handled_keys():
    """Every ord('x') the live loop acts on."""

    return set(re.findall(r"ord\('(\w)'\)", MAIN.read_text(encoding="utf-8")))


class KeyHintTest(unittest.TestCase):

    def test_the_loop_handles_the_keys_we_expect(self):
        """Guards the regex itself: a rewrite that finds nothing passes
        the next test vacuously."""

        self.assertEqual(handled_keys(), {"q", "l", "c", "s", "w"})

    def test_every_handled_key_is_offered_to_the_driver(self):
        hinted = set(re.findall(r"\b([A-Z])\s", KEY_HINTS))

        for key in handled_keys():
            self.assertIn(
                key.upper(),
                hinted,
                f"the loop handles {key.upper()} but the panel never "
                f"says so: {KEY_HINTS!r}"
            )

    def test_the_hints_fit_the_panel(self):
        self.assertLessEqual(
            text_width(KEY_HINTS, 0.42), PANEL_WIDTH - 2 * LABEL_X
        )


def state(context):
    return {"risk": {"context": context}}


class WeatherRowTest(unittest.TestCase):

    def row(self, context):
        rows = DashboardRenderer._weather_rows(state(context))

        return rows[0] if rows else None

    def test_nothing_is_shown_when_no_weather_is_in_play(self):
        self.assertIsNone(
            self.row(DrivingContext(weather="clear",
                                    weather_source="default"))
        )

    def test_the_condition_and_its_source_are_shown(self):
        label, value, _color, detail = self.row(
            DrivingContext(weather="heavy_rain", weather_source="forecast")
        )

        self.assertEqual(label, "WEATHER")
        self.assertEqual(value, "HEAVY RAIN")
        self.assertEqual(detail, "forecast")

    def test_a_driver_override_is_named_as_one(self):
        _, value, _color, detail = self.row(
            DrivingContext(weather="ice", weather_source="driver")
        )

        self.assertEqual(value, "ICE")
        self.assertEqual(detail, "driver")

    def test_a_neutral_forecast_still_shows(self):
        """
        An honest 1.0 and an ignorant one weigh the same, so the row has
        to stay up whenever a forecast is involved - otherwise a driver
        cannot tell the forecast has gone stale and needs overruling.
        """

        row = self.row(
            DrivingContext(weather="clear",
                           weather_source="forecast (stale)")
        )

        self.assertIsNotNone(row)
        self.assertEqual(row[3], "forecast (stale)")

    def test_worse_weather_reads_more_urgently(self):
        def colour(weather):
            return self.row(
                DrivingContext(weather=weather, weather_source="driver")
            )[2]

        self.assertNotEqual(colour("rain"), colour("snow"))
        self.assertNotEqual(colour("clear"), colour("rain"))

    def test_a_missing_context_is_not_an_error(self):
        self.assertEqual(DashboardRenderer._weather_rows(state(None)), [])


if __name__ == "__main__":
    unittest.main()

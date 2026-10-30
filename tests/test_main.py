"""
The entry point's argument handling and its failure to start.

Nothing here opens a camera or a window: these cover the decisions
main() makes before the loop, and the one it makes when the loop can
never begin.
"""

import io
import unittest
from contextlib import redirect_stderr
from unittest import mock

from app.main import main, parse_args, parse_source
from app.config import WEATHER_MULTIPLIERS


def args(*argv):
    with mock.patch("sys.argv", ["main.py", *argv]):
        return parse_args()


class ParseSourceTest(unittest.TestCase):

    def test_a_digit_is_a_camera_index(self):
        self.assertEqual(parse_source("0"), 0)
        self.assertEqual(parse_source("1"), 1)

    def test_anything_else_is_a_path(self):
        self.assertEqual(parse_source("drive.mp4"), "drive.mp4")


class ArgumentTest(unittest.TestCase):

    def test_weather_is_undeclared_by_default(self):
        """
        None rather than "clear" is what lets the forecast speak; a
        default of "clear" would silently overrule it every run.
        """

        self.assertIsNone(args().weather)

    def test_a_declared_weather_is_kept(self):
        self.assertEqual(args("--weather", "snow").weather, "snow")

    def test_every_weighted_condition_can_be_declared(self):
        for weather in WEATHER_MULTIPLIERS:
            self.assertEqual(args("--weather", weather).weather, weather)

    def test_an_unknown_weather_is_refused(self):
        with self.assertRaises(SystemExit), redirect_stderr(io.StringIO()):
            args("--weather", "sleet")

    def test_the_forecast_is_off_without_coordinates(self):
        parsed = args()

        self.assertIsNone(parsed.latitude)
        self.assertIsNone(parsed.longitude)

    def test_coordinates_can_be_given(self):
        parsed = args("--latitude", "28.6", "--longitude", "77.2")

        self.assertAlmostEqual(parsed.latitude, 28.6)
        self.assertAlmostEqual(parsed.longitude, 77.2)

    def test_the_toggles_default_to_on(self):
        parsed = args()

        self.assertFalse(parsed.no_forecast)
        self.assertFalse(parsed.no_enhance)
        self.assertFalse(parsed.static_weights)
        self.assertFalse(parsed.mute)


class UnopenableSourceTest(unittest.TestCase):
    """
    A mistyped --source used to end in a traceback, while the same
    mistake in the evaluation harness printed one line. Same mistake,
    same treatment.
    """

    def run_with_source(self, source):
        stderr = io.StringIO()

        with mock.patch("sys.argv", ["main.py", "--source", source]):
            with redirect_stderr(stderr):
                code = main()

        return code, stderr.getvalue()

    def test_a_missing_file_exits_non_zero(self):
        code, _ = self.run_with_source("no/such/file.mp4")

        self.assertEqual(code, 1)

    def test_a_missing_file_explains_itself_in_one_line(self):
        _, message = self.run_with_source("no/such/file.mp4")

        self.assertIn("no/such/file.mp4", message)
        self.assertIn("Unable to open video source", message)
        self.assertNotIn("Traceback", message)
        self.assertEqual(len(message.strip().splitlines()), 1)

    def test_nothing_is_logged_when_the_camera_never_opens(self):
        """
        Failing to start must not leave an empty session behind.
        """

        with mock.patch("app.main.Database") as database:
            self.run_with_source("no/such/file.mp4")

        database.assert_not_called()


if __name__ == "__main__":
    unittest.main()

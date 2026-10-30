"""
Runs the Streamlit analytics dashboard headless with Streamlit's AppTest.
"""

import os
import sys
import tempfile
import unittest
from unittest import mock

from streamlit.testing.v1 import AppTest

from database.database import Database
from database.session_logger import SessionLogger
from tests.test_database import make_state

DASHBOARD = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "dashboard", "dashboard.py"
)


def run_dashboard(db_path):

    app = AppTest.from_file(DASHBOARD, default_timeout=60)

    # The dashboard reads --db from the command line.
    with mock.patch.object(sys, "argv", ["dashboard.py", "--db", db_path]):
        app.run()

    return app


class DashboardTest(unittest.TestCase):

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.db_path = os.path.join(self.folder.name, "sessions.db")

    def record_session(self, context=None, dynamic=False,
                       wheel=None, camera=None):

        database = Database(self.db_path)
        logger = SessionLogger(database, "0")

        t = 0.0

        for i in range(300):
            events = []

            if i == 120:
                events = [{"type": "PHONE", "severity": "high",
                           "message": "PHONE IN USE"}]

            face = not 200 <= i < 240
            score = 60.0 if 120 <= i < 180 else 100.0

            logger.log(make_state(t, score, events, blinks=i // 60,
                                  face=face, context=context,
                                  dynamic=dynamic,
                                  brightness=30.0 if context else None,
                                  wheel=wheel(i) if wheel else None,
                                  camera=camera(i) if camera else None))
            t += 1 / 30

        logger.close()
        database.close()

    def test_missing_database(self):
        app = run_dashboard(self.db_path)

        self.assertFalse(app.exception)
        self.assertIn("No sessions recorded yet", app.info[0].value)
        # Viewing must not create the database.
        self.assertFalse(os.path.exists(self.db_path))

    def test_recorded_session(self):
        self.record_session()

        app = run_dashboard(self.db_path)

        self.assertFalse(app.exception, app.exception)

        values = {metric.label: metric.value for metric in app.metric}

        self.assertEqual(values["Duration"], "0:10")
        self.assertEqual(values["Lowest safety index"], "60")
        self.assertEqual(values["Blinks"], "4")
        # make_state() labels the 60-point stretch (2 s of 10) DANGER.
        self.assertEqual(values["Time in danger"], "20%")

        self.assertEqual(len(app.selectbox), 1)
        self.assertIn("camera 0", app.selectbox[0].options[0])

    def test_a_static_session_says_no_weighting_was_applied(self):
        self.record_session()

        app = run_dashboard(self.db_path)
        captions = " ".join(c.value for c in app.caption)

        self.assertFalse(app.exception, app.exception)
        self.assertIn("static", captions)
        self.assertIn("flat penalty table", captions)

    def test_a_dynamic_session_charts_its_weighting(self):
        from risk_engine.context import DrivingContext

        self.record_session(
            context=DrivingContext(hour=3, elapsed=3600, brightness=30.0),
            dynamic=True,
        )

        app = run_dashboard(self.db_path)

        self.assertFalse(app.exception, app.exception)

        headings = [s.value for s in app.subheader]
        self.assertIn("Risk weighting", headings)

        captions = " ".join(c.value for c in app.caption)
        self.assertIn("dynamic", captions)
        self.assertNotIn("flat penalty table", captions)

    def test_a_forecast_session_names_the_weather_and_its_source(self):
        from risk_engine.context import DrivingContext

        self.record_session(
            context=DrivingContext(
                hour=3,
                elapsed=3600,
                brightness=30.0,
                weather="rain",
                weather_source="forecast",
            ),
            dynamic=True,
        )

        app = run_dashboard(self.db_path)
        captions = " ".join(c.value for c in app.caption)

        self.assertFalse(app.exception, app.exception)
        self.assertIn("rain", captions)
        self.assertIn("forecast", captions)

    def test_a_declared_session_says_the_driver_set_it(self):
        """
        The audit case. Reviewing a session is the only place the
        difference between a forecast and an assertion can be checked,
        so the dashboard has to carry it.
        """

        from risk_engine.context import DrivingContext

        self.record_session(
            context=DrivingContext(
                hour=12,
                brightness=120.0,
                weather="snow",
                weather_source="driver",
            ),
            dynamic=True,
        )

        app = run_dashboard(self.db_path)
        captions = " ".join(c.value for c in app.caption)

        self.assertFalse(app.exception, app.exception)
        self.assertIn("snow", captions)
        self.assertIn("driver", captions)

    def test_a_session_recorded_before_weather_existed_still_opens(self):
        """
        The old databases have no weather column at all, and the
        dashboard must not insist on one.
        """

        self.record_session()

        app = run_dashboard(self.db_path)
        captions = " ".join(c.value for c in app.caption)

        self.assertFalse(app.exception, app.exception)
        self.assertNotIn("Weather:", captions)


    def test_a_session_with_the_wheel_in_view_charts_it(self):
        """
        The live panel carried hands-on-wheel from the start; until the
        log did too, a drive could not be reviewed for it afterwards.
        """

        def wheel(i):
            off = 150 <= i < 240

            return {
                "status": "NONE" if off else "BOTH",
                "count": 0 if off else 2,
                "hands_off": off,
                "hands_off_count": 1 if i >= 150 else 0,
                "missing_for": 4.0 if off else 0.0,
                "can_see": True,
            }

        self.record_session(wheel=wheel)

        app = run_dashboard(self.db_path)

        self.assertFalse(app.exception, app.exception)

        headings = [s.value for s in app.subheader]
        self.assertIn("Camera and wheel", headings)

        captions = " ".join(c.value for c in app.caption)
        self.assertIn("1 hands-off episode", captions)

    def test_an_unseen_wheel_says_so_rather_than_reading_as_clean(self):
        """
        The failure this feature is most likely to hit: a camera framed
        on the face, where no hands-off episode means the detector
        declined to judge. A dashboard that showed a bare zero would
        read as a flawless drive.
        """

        def wheel(i):
            return {
                "status": "UNKNOWN", "count": 0, "hands_off": False,
                "hands_off_count": 0, "missing_for": 0.0,
                "can_see": False,
            }

        self.record_session(wheel=wheel)

        app = run_dashboard(self.db_path)
        captions = " ".join(c.value for c in app.caption)

        self.assertFalse(app.exception, app.exception)
        self.assertIn("never in the camera's view", captions)
        self.assertIn("not a clean drive", captions)

    def test_a_blocked_lens_is_reported_as_a_share_of_the_session(self):
        def camera(i):
            blocked = i >= 240

            return {
                "blocked": blocked,
                "blocked_count": 1 if blocked else 0,
                "brightness": 2.0 if blocked else 90.0,
                "variance": 1.0 if blocked else 40.0,
                "blocked_duration": 3.0 if blocked else 0.0,
            }

        self.record_session(camera=camera)

        app = run_dashboard(self.db_path)

        self.assertFalse(app.exception, app.exception)

        values = {metric.label: metric.value for metric in app.metric}

        # The last 2 s of a 10 s session.
        self.assertEqual(values["Lens blocked"], "20%")

        captions = " ".join(c.value for c in app.caption)
        self.assertIn("1 spell", captions)

    def test_a_session_recorded_before_either_monitor_still_opens(self):
        self.record_session()

        app = run_dashboard(self.db_path)

        self.assertFalse(app.exception, app.exception)

        headings = [s.value for s in app.subheader]
        self.assertNotIn("Camera and wheel", headings)


if __name__ == "__main__":
    unittest.main()

import os
import sqlite3
import tempfile
import unittest

import numpy as np
import pandas as pd

from database.database import Database
from database.session_logger import SessionLogger


def make_state(timestamp, score=100.0, events=(), blinks=0, face=True,
               context=None, dynamic=False, brightness=None,
               wheel=None, camera=None):

    state = {
        "timestamp": timestamp,
        "fps": 20.0,
        "brightness": brightness,
        "face_detected": face,
        "blink": None,
        "drowsiness": None,
        "yawn": None,
        "head_pose": None,
        "gaze": None,
        "distraction": None,
        "phone": {"in_use": False, "usage_count": 0},
        "seatbelt": {"status": "UNKNOWN"},
        # Left out by default, so the tests for sessions
        # recorded before these monitors existed keep working.
        "wheel": wheel,
        "camera": camera,
        "emotion": None,
        "risk": {
            "score": score,
            "level": "SAFE" if score >= 75 else "DANGER",
            "events": list(events),
            "dynamic": dynamic,
            "context": context,
        },
    }

    if face:
        state["blink"] = {
            # NumPy scalars, as the real detectors produce.
            "ear": np.float64(0.28),
            "blink_rate": 12.0,
            "blink_count": np.int64(blinks),
            "threshold": 0.24,
        }
        state["head_pose"] = {"yaw": -3.0, "pitch": 2.0, "roll": 0.5,
                              "direction": "FORWARD", "calibrated": True}
        state["distraction"] = {"eyes_on_road": True,
                                "distraction_count": 0}
        state["emotion"] = {"emotion": "Neutral", "confidence": 0.9}

    return state


class DatabaseTest(unittest.TestCase):

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)

        self.path = os.path.join(self.folder.name, "nested", "test.db")
        self.database = Database(self.path)
        self.addCleanup(self.database.close)

    def run_session(self, seconds=5.0, fps=30.0):

        logger = SessionLogger(self.database, "0")
        t = 100.0

        for i in range(int(seconds * fps)):
            events = []

            if i == 45:
                events = [{"type": "PHONE", "severity": "high",
                           "message": "PHONE IN USE"}]

            score = 60.0 if 45 <= i < 90 else 100.0
            logger.log(make_state(t, score, events, blinks=i // 30))
            t += 1 / fps

        logger.close()

        return logger.session_id

    def summary(self, session_id):
        """The sessions row for one session."""

        sessions = self.database.sessions()

        return sessions[sessions["id"] == session_id].iloc[0]

    def test_creates_the_folder(self):
        self.assertTrue(os.path.exists(self.path))

    def test_weighting_columns_round_trip(self):
        from risk_engine.context import DrivingContext

        logger = SessionLogger(self.database, "0")
        context = DrivingContext(hour=3, elapsed=3600, brightness=15.0)

        for i in range(60):
            logger.log(make_state(100.0 + i / 30.0, context=context,
                                  dynamic=True, brightness=15.0))

        logger.close()

        metrics = self.database.metrics(logger.session_id)
        row = metrics.iloc[0]

        self.assertEqual(row["weights_mode"], "dynamic")
        self.assertAlmostEqual(row["brightness"], 15.0)
        self.assertAlmostEqual(row["ear_threshold"], 0.24)
        self.assertAlmostEqual(row["context_fatigue"],
                               context.fatigue_multiplier())
        self.assertAlmostEqual(row["context_attention"],
                               context.attention_multiplier())

    def test_static_sessions_say_so(self):
        session_id = self.run_session(seconds=2.0)
        metrics = self.database.metrics(session_id)

        self.assertEqual(metrics["weights_mode"].iloc[0], "static")
        self.assertTrue(metrics["context_fatigue"].isna().all())

    def test_a_database_from_an_older_version_is_migrated(self):
        """
        CREATE TABLE IF NOT EXISTS leaves an existing database on its
        old schema, so the added columns have to be filled in by hand.
        """

        path = os.path.join(self.folder.name, "old.db")

        old = sqlite3.connect(path)
        old.executescript("""
            CREATE TABLE sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                started_at TEXT NOT NULL, ended_at TEXT, source TEXT,
                duration REAL, average_score REAL, minimum_score REAL,
                blinks INTEGER, yawns INTEGER, microsleeps INTEGER,
                distractions INTEGER, phone_uses INTEGER);
            CREATE TABLE metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL, recorded_at TEXT NOT NULL,
                elapsed REAL NOT NULL, safety_score REAL);
            CREATE TABLE events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL, recorded_at TEXT NOT NULL,
                elapsed REAL NOT NULL, event_type TEXT NOT NULL,
                severity TEXT NOT NULL, message TEXT);
            INSERT INTO sessions (started_at, source) VALUES ('t0', '0');
            INSERT INTO metrics (session_id, recorded_at, elapsed,
                safety_score) VALUES (1, 't0', 0.0, 88.0);
        """)
        old.commit()
        old.close()

        database = Database(path)
        self.addCleanup(database.close)

        columns = {
            row[1] for row in
            database.connection.execute("PRAGMA table_info(metrics)")
        }

        self.assertIn("weights_mode", columns)
        self.assertIn("context_fatigue", columns)
        self.assertIn("hands_status", columns)
        self.assertIn("camera_blocked", columns)

        # The summary columns are added to sessions in the same
        # pass; before this they were only ever in CREATE TABLE.
        session_columns = {
            row[1] for row in
            database.connection.execute("PRAGMA table_info(sessions)")
        }

        self.assertIn("hands_off_episodes", session_columns)
        self.assertIn("camera_blocks", session_columns)

        # The rows that were already there survive.
        metrics = database.metrics(1)
        self.assertEqual(len(metrics), 1)
        self.assertAlmostEqual(metrics["safety_score"].iloc[0], 88.0)

    def test_the_wheel_and_the_lens_are_logged(self):
        """
        The live panel showed both from the day they were written; the
        session log did not, so a drive could not be reviewed for
        either afterwards.
        """

        logger = SessionLogger(self.database, "0")

        t = 100.0

        for i in range(90):
            hands_off = 30 <= i < 60

            logger.log(make_state(
                t,
                wheel={
                    "status": "NONE" if hands_off else "BOTH",
                    "count": 0 if hands_off else 2,
                    "hands_off": hands_off,
                    "hands_off_count": 1 if i >= 30 else 0,
                    "missing_for": 5.0 if hands_off else 0.0,
                    "can_see": True,
                },
                camera={
                    "blocked": i >= 60,
                    "blocked_count": 1 if i >= 60 else 0,
                    "brightness": 2.0 if i >= 60 else 90.0,
                    "variance": 1.0 if i >= 60 else 40.0,
                    "blocked_duration": 3.0 if i >= 60 else 0.0,
                },
            ))
            t += 1 / 30.0

        logger.close()

        metrics = self.database.metrics(logger.session_id)

        self.assertEqual(list(metrics["hands_status"]),
                         ["BOTH", "NONE", "BOTH"])
        self.assertEqual(
            [bool(v) for v in metrics["camera_blocked"]],
            [False, False, True]
        )

        session = self.summary(logger.session_id)

        self.assertEqual(session["hands_off_episodes"], 1)
        self.assertEqual(session["camera_blocks"], 1)

    def test_an_unseen_wheel_is_logged_as_unknown_not_as_nothing(self):
        """
        UNKNOWN is a reading. Logging it as NULL would leave a
        reviewer unable to tell a wheel that was out of shot from a
        wheel that was watched all drive and never let go of.
        """

        logger = SessionLogger(self.database, "0")

        t = 100.0

        for _ in range(60):
            logger.log(make_state(t, wheel={
                "status": "UNKNOWN", "count": 0, "hands_off": False,
                "hands_off_count": 0, "missing_for": 0.0,
                "can_see": False,
            }))
            t += 1 / 30.0

        logger.close()

        metrics = self.database.metrics(logger.session_id)

        self.assertTrue((metrics["hands_status"] == "UNKNOWN").all())
        self.assertEqual(
            self.summary(logger.session_id)["hands_off_episodes"], 0
        )

    def test_a_session_without_either_monitor_logs_nulls(self):
        """
        Not zeros. A session recorded before these existed must not
        claim the lens was clear when nobody was watching it.
        """

        session_id = self.run_session(seconds=2.0)
        metrics = self.database.metrics(session_id)

        self.assertTrue(metrics["hands_status"].isna().all())
        self.assertTrue(metrics["camera_blocked"].isna().all())

    def test_metrics_are_sampled_once_per_second(self):
        session_id = self.run_session(seconds=5.0)
        metrics = self.database.metrics(session_id)

        self.assertEqual(len(metrics), 5)

        # On a fixed one-second grid: no drift from frame timing.
        for i, elapsed in enumerate(metrics["elapsed"]):
            self.assertAlmostEqual(elapsed, float(i), delta=1 / 30)

        self.assertAlmostEqual(metrics["ear"].iloc[0], 0.28)
        self.assertEqual(metrics["head_direction"].iloc[0], "FORWARD")
        self.assertEqual(metrics["emotion"].iloc[0], "Neutral")

    def test_events_are_stored(self):
        session_id = self.run_session()
        events = self.database.events(session_id)

        self.assertEqual(list(events["event_type"]), ["PHONE"])
        self.assertAlmostEqual(events["elapsed"].iloc[0], 1.5, places=2)
        self.assertEqual(events["severity"].iloc[0], "high")

    def test_session_summary(self):
        session_id = self.run_session(seconds=5.0)
        sessions = self.database.sessions()
        row = sessions[sessions["id"] == session_id].iloc[0]

        self.assertEqual(row["source"], "0")
        self.assertAlmostEqual(row["duration"], 5.0, delta=0.1)
        self.assertEqual(row["blinks"], 4)
        self.assertAlmostEqual(row["minimum_score"], 60.0)
        # Snapshots at 0-4 s; the one at ~2 s falls inside the dip.
        self.assertAlmostEqual(row["average_score"], 92.0)
        self.assertIsNotNone(row["ended_at"])

    def test_frames_without_face_store_nulls(self):
        logger = SessionLogger(self.database, "video.mp4")
        logger.log(make_state(0.0, face=False))
        logger.close()

        metrics = self.database.metrics(logger.session_id)

        self.assertEqual(metrics["face_detected"].iloc[0], 0)
        self.assertTrue(pd.isna(metrics["ear"].iloc[0]))

    def test_head_pose_is_not_logged_while_calibrating(self):
        state = make_state(0.0)
        state["head_pose"] = dict(state["head_pose"], calibrated=False)

        logger = SessionLogger(self.database, "0")
        logger.log(state)
        logger.close()

        metrics = self.database.metrics(logger.session_id)

        self.assertTrue(pd.isna(metrics["yaw"].iloc[0]))
        self.assertTrue(pd.isna(metrics["head_direction"].iloc[0]))
        # Everything else is still logged.
        self.assertAlmostEqual(metrics["ear"].iloc[0], 0.28)

    def test_sessions_are_listed_newest_first(self):
        first = self.run_session(seconds=1.0)
        second = self.run_session(seconds=1.0)

        self.assertEqual(list(self.database.sessions()["id"]),
                         [second, first])

    def test_long_session_does_not_drift(self):
        session_id = self.run_session(seconds=60.0)
        self.assertEqual(len(self.database.metrics(session_id)), 60)

    def test_data_survives_reopening(self):
        session_id = self.run_session(seconds=2.0)
        self.database.close()

        reopened = Database(self.path)
        self.addCleanup(reopened.close)
        self.database = reopened

        self.assertEqual(len(reopened.metrics(session_id)), 2)


if __name__ == "__main__":
    unittest.main()

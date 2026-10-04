import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from database.database import Database
from database.session_logger import SessionLogger


def make_state(timestamp, score=100.0, events=(), blinks=0, face=True):

    state = {
        "timestamp": timestamp,
        "fps": 20.0,
        "face_detected": face,
        "blink": None,
        "drowsiness": None,
        "yawn": None,
        "head_pose": None,
        "gaze": None,
        "distraction": None,
        "phone": {"in_use": False, "usage_count": 0},
        "seatbelt": {"status": "UNKNOWN"},
        "emotion": None,
        "risk": {
            "score": score,
            "level": "SAFE" if score >= 75 else "DANGER",
            "events": list(events),
        },
    }

    if face:
        state["blink"] = {
            # NumPy scalars, as the real detectors produce.
            "ear": np.float64(0.28),
            "blink_rate": 12.0,
            "blink_count": np.int64(blinks),
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

    def test_creates_the_folder(self):
        self.assertTrue(os.path.exists(self.path))

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

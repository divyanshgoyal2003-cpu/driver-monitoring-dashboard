"""
Session storage in SQLite.

Each run of the monitor is a session. While it runs, a snapshot of the
measurements is stored about once per second (metrics), every event as
it happens (events), and a summary when it ends (sessions). The
analytics dashboard reads these tables.
"""

import os
import sqlite3

import pandas as pd

from app.config import DATABASE_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    source TEXT,
    duration REAL,
    average_score REAL,
    minimum_score REAL,
    blinks INTEGER,
    yawns INTEGER,
    microsleeps INTEGER,
    distractions INTEGER,
    phone_uses INTEGER
);

CREATE TABLE IF NOT EXISTS metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES sessions(id),
    recorded_at TEXT NOT NULL,
    elapsed REAL NOT NULL,
    fps REAL,
    face_detected INTEGER,
    ear REAL,
    blink_rate REAL,
    perclos REAL,
    drowsiness TEXT,
    mar REAL,
    yaw REAL,
    pitch REAL,
    roll REAL,
    head_direction TEXT,
    gaze_direction TEXT,
    eyes_on_road INTEGER,
    phone_in_use INTEGER,
    seatbelt TEXT,
    emotion TEXT,
    emotion_confidence REAL,
    safety_score REAL,
    risk_level TEXT,
    brightness REAL,
    ear_threshold REAL,
    context_fatigue REAL,
    context_attention REAL,
    weights_mode TEXT,
    weather TEXT,
    weather_source TEXT
);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES sessions(id),
    recorded_at TEXT NOT NULL,
    elapsed REAL NOT NULL,
    event_type TEXT NOT NULL,
    severity TEXT NOT NULL,
    message TEXT
);

CREATE INDEX IF NOT EXISTS idx_metrics_session ON metrics(session_id);
CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id);
"""

METRIC_COLUMNS = (
    "recorded_at", "elapsed", "fps", "face_detected", "ear", "blink_rate",
    "perclos", "drowsiness", "mar", "yaw", "pitch", "roll",
    "head_direction", "gaze_direction", "eyes_on_road", "phone_in_use",
    "seatbelt", "emotion", "emotion_confidence", "safety_score",
    "risk_level", "brightness", "ear_threshold", "context_fatigue",
    "context_attention", "weights_mode", "weather", "weather_source"
)

# Columns added after the first release. CREATE TABLE IF NOT EXISTS
# leaves an existing database on the old schema, so they are added by
# hand when a database recorded by an earlier version is opened.
ADDED_COLUMNS = {
    "metrics": (
        ("brightness", "REAL"),
        ("ear_threshold", "REAL"),
        ("context_fatigue", "REAL"),
        ("context_attention", "REAL"),
        ("weights_mode", "TEXT"),
        ("weather", "TEXT"),
        ("weather_source", "TEXT"),
    ),
}

SUMMARY_COLUMNS = (
    "ended_at", "duration", "average_score", "minimum_score", "blinks",
    "yawns", "microsleeps", "distractions", "phone_uses"
)


class Database:

    def __init__(self, path=DATABASE_PATH):

        folder = os.path.dirname(path)

        if folder:
            os.makedirs(folder, exist_ok=True)

        self.path = path
        self.closed = False
        self.connection = sqlite3.connect(path)
        self.connection.executescript(SCHEMA)
        self._add_missing_columns()
        self.connection.commit()

    def _add_missing_columns(self):
        """Bring a database written by an earlier version up to date."""

        for table, columns in ADDED_COLUMNS.items():

            existing = {
                row[1] for row in
                self.connection.execute(f"PRAGMA table_info({table})")
            }

            for name, column_type in columns:
                if name not in existing:
                    self.connection.execute(
                        f"ALTER TABLE {table} ADD COLUMN {name} "
                        f"{column_type}"
                    )

    # ----- writing -----

    def start_session(self, started_at, source):

        cursor = self.connection.execute(
            "INSERT INTO sessions (started_at, source) VALUES (?, ?)",
            (started_at, source)
        )
        self.connection.commit()

        return cursor.lastrowid

    def add_metrics(self, session_id, metrics):

        columns = ", ".join(("session_id",) + METRIC_COLUMNS)
        placeholders = ", ".join("?" * (len(METRIC_COLUMNS) + 1))

        self.connection.execute(
            f"INSERT INTO metrics ({columns}) VALUES ({placeholders})",
            (session_id,) + tuple(metrics.get(c) for c in METRIC_COLUMNS)
        )

    def add_event(self, session_id, recorded_at, elapsed, event_type,
                  severity, message):

        self.connection.execute(
            "INSERT INTO events (session_id, recorded_at, elapsed, "
            "event_type, severity, message) VALUES (?, ?, ?, ?, ?, ?)",
            (session_id, recorded_at, elapsed, event_type, severity,
             message)
        )

    def end_session(self, session_id, summary):

        assignments = ", ".join(f"{c} = ?" for c in SUMMARY_COLUMNS)

        self.connection.execute(
            f"UPDATE sessions SET {assignments} WHERE id = ?",
            tuple(summary.get(c) for c in SUMMARY_COLUMNS) + (session_id,)
        )
        self.connection.commit()

    def commit(self):
        self.connection.commit()

    def close(self):

        if self.closed:
            return

        self.connection.commit()
        self.connection.close()
        self.closed = True

    # ----- reading (analytics dashboard) -----

    def sessions(self):

        return pd.read_sql_query(
            "SELECT * FROM sessions ORDER BY id DESC",
            self.connection
        )

    def metrics(self, session_id):

        return pd.read_sql_query(
            "SELECT * FROM metrics WHERE session_id = ? ORDER BY elapsed",
            self.connection,
            params=(session_id,)
        )

    def events(self, session_id):

        return pd.read_sql_query(
            "SELECT * FROM events WHERE session_id = ? ORDER BY elapsed",
            self.connection,
            params=(session_id,)
        )

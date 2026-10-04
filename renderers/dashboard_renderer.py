import cv2
import numpy as np

from app.config import DISPLAY_HEIGHT, PANEL_WIDTH, SHOW_FPS
from renderers.theme import (
    BACKGROUND,
    DIVIDER,
    TEXT,
    MUTED,
    GOOD,
    WARN,
    BAD,
    put_text,
    put_text_right
)

ROW_HEIGHT = 27
LABEL_X = 16
VALUE_X = 120

MISSING = "--"


class DashboardRenderer:
    """
    Builds the status panel and attaches it to the right of the camera
    frame. Each feature contributes rows of (label, value, colour,
    detail); a None in the row list draws a divider.
    """

    def __init__(self, panel_width=PANEL_WIDTH,
                 display_height=DISPLAY_HEIGHT):

        self.panel_width = panel_width
        self.display_height = display_height

    def draw(self, frame, state):

        display = self._fit_height(frame)

        panel = np.full(
            (self.display_height, self.panel_width, 3),
            BACKGROUND,
            dtype=np.uint8
        )

        y = self._draw_header(panel, state)

        y = self._draw_rows(panel, y, self._rows(state))

        self._draw_footer(panel)

        return np.hstack([display, panel])

    def _fit_height(self, frame):

        h, w = frame.shape[:2]

        if h == self.display_height:
            return frame

        scale = self.display_height / h

        return cv2.resize(
            frame,
            (int(round(w * scale)), self.display_height),
            interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        )

    def _draw_header(self, panel, state):

        put_text(panel, "DRIVER MONITOR", (LABEL_X, 34), 0.75, TEXT, 2)

        if SHOW_FPS:
            put_text_right(
                panel,
                f"{state['fps']:.1f} FPS",
                self.panel_width - LABEL_X,
                34,
                0.5,
                MUTED
            )

        y = 50
        self._divider(panel, y)

        return y + 8

    def _rows(self, state):

        rows = []

        rows.extend(self._face_rows(state))
        rows.append(None)
        rows.extend(self._eye_rows(state["blink"]))
        rows.extend(self._gaze_rows(state["gaze"]))
        rows.extend(self._drowsiness_rows(state["drowsiness"]))
        rows.append(None)
        rows.extend(self._yawn_rows(state["yawn"]))
        rows.append(None)
        rows.extend(self._head_pose_rows(state["head_pose"]))
        rows.extend(self._attention_rows(state["distraction"]))
        rows.append(None)
        rows.extend(self._phone_rows(state["phone"]))
        rows.extend(self._emotion_rows(state["emotion"]))
        rows.extend(self._seatbelt_rows(state["seatbelt"]))

        return rows

    def _face_rows(self, state):

        count = state["face_count"]
        driver_id = state["driver_id"]

        detail = ""

        if driver_id is not None:
            detail = f"driver ID {driver_id}"
        elif state["face_detected"]:
            detail = "driver found"

        return [
            ("FACES", str(count), TEXT if count else WARN, detail)
        ]

    def _eye_rows(self, blink):

        if not blink:
            return [
                ("EYES", MISSING, MUTED, ""),
                ("BLINKS", MISSING, MUTED, "")
            ]

        closed = blink["eyes_closed"]

        return [
            (
                "EYES",
                "CLOSED" if closed else "OPEN",
                BAD if closed else GOOD,
                f"EAR {blink['ear']:.2f}"
            ),
            (
                "BLINKS",
                str(blink["blink_count"]),
                TEXT,
                f"{blink['blink_rate']:.0f} / min"
            )
        ]

    def _gaze_rows(self, gaze):

        if not gaze:
            return [("GAZE", MISSING, MUTED, "")]

        direction = gaze["direction"]

        return [
            (
                "GAZE",
                direction,
                GOOD if direction == "CENTER" else WARN,
                f"yaw {gaze['gaze_yaw']:+.0f}"
            )
        ]

    def _drowsiness_rows(self, drowsiness):

        if not drowsiness:
            return [("DROWSINESS", MISSING, MUTED, "")]

        state = drowsiness["state"]

        color = {
            "AWAKE": GOOD,
            "TIRED": WARN,
            "DROWSY": BAD,
            "MICROSLEEP": BAD
        }[state]

        detail = f"PERCLOS {drowsiness['perclos'] * 100:.0f}%"

        if not drowsiness["perclos_ready"]:
            detail += " (warming up)"

        return [("DROWSINESS", state, color, detail)]

    def _yawn_rows(self, yawn):

        if not yawn:
            return [
                ("MOUTH", MISSING, MUTED, ""),
                ("YAWNS", MISSING, MUTED, "")
            ]

        return [
            (
                "MOUTH",
                "OPEN" if yawn["mouth_open"] else "CLOSED",
                WARN if yawn["mouth_open"] else GOOD,
                f"MAR {yawn['mar']:.2f}"
            ),
            (
                "YAWNS",
                str(yawn["yawn_count"]),
                WARN if yawn["recent_yawns"] >= 3 else TEXT,
                f"{yawn['recent_yawns']} in last 5 min"
            )
        ]

    def _head_pose_rows(self, head_pose):

        if not head_pose:
            return [
                ("HEAD", MISSING, MUTED, ""),
                ("ANGLES", MISSING, MUTED, "")
            ]

        direction = head_pose["direction"]

        if not head_pose["calibrated"]:
            value, color = "CALIBRATING", WARN
        else:
            value = direction
            color = GOOD if direction == "FORWARD" else WARN

        return [
            ("HEAD", value, color, ""),
            (
                "ANGLES",
                f"Y {head_pose['yaw']:+.0f}  "
                f"P {head_pose['pitch']:+.0f}  "
                f"R {head_pose['roll']:+.0f}",
                TEXT,
                "deg"
            )
        ]

    def _attention_rows(self, distraction):

        if not distraction:
            return [("ATTENTION", MISSING, MUTED, "")]

        count = distraction["distraction_count"]
        detail = f"{count} distraction{'' if count == 1 else 's'}"

        if distraction["distracted"]:
            return [("ATTENTION", "DISTRACTED", BAD, detail)]

        if not distraction["eyes_on_road"]:
            seconds = distraction["off_road_duration"]
            return [("ATTENTION", f"OFF ROAD {seconds:.1f}s", WARN, detail)]

        return [("ATTENTION", "ON ROAD", GOOD, detail)]

    def _phone_rows(self, phone):

        count = phone["usage_count"]
        detail = f"{count} use{'' if count == 1 else 's'}"

        if phone["in_use"]:
            seconds = phone["usage_duration"]
            return [("PHONE", f"IN USE {seconds:.0f}s", BAD, detail)]

        if phone["detected"]:
            return [("PHONE", "DETECTED", WARN, detail)]

        return [("PHONE", "NOT DETECTED", GOOD, detail)]

    def _emotion_rows(self, emotion):

        if not emotion:
            return [("EMOTION", MISSING, MUTED, "")]

        return [
            (
                "EMOTION",
                emotion["emotion"].upper(),
                WARN if emotion["negative"] else GOOD,
                f"{emotion['confidence'] * 100:.0f}%"
            )
        ]

    def _seatbelt_rows(self, seatbelt):

        status = seatbelt["status"]

        if status == "ON":
            return [("SEATBELT", "ON", GOOD, "")]

        if status == "OFF":
            return [("SEATBELT", "NOT DETECTED", WARN, "")]

        return [("SEATBELT", "UNKNOWN", MUTED, "chest not in view")]

    def _draw_rows(self, panel, y, rows):

        right_x = self.panel_width - LABEL_X

        for row in rows:

            if row is None:
                self._divider(panel, y + 4)
                y += 10
                continue

            label, value, color, detail = row

            y += ROW_HEIGHT

            put_text(panel, label, (LABEL_X, y), 0.45, MUTED)
            put_text(panel, value, (VALUE_X, y), 0.55, color, 1)

            if detail:
                put_text_right(panel, detail, right_x, y, 0.45, MUTED)

        return y

    def _draw_footer(self, panel):

        y = self.display_height - 14

        self._divider(panel, y - 20)

        put_text(
            panel,
            "Q quit   L landmarks   C calibrate",
            (LABEL_X, y),
            0.42,
            MUTED
        )

    def _divider(self, panel, y):

        cv2.line(
            panel,
            (LABEL_X, y),
            (self.panel_width - LABEL_X, y),
            DIVIDER,
            1
        )

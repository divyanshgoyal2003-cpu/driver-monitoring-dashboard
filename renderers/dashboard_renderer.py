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
    LEVEL_COLORS,
    SEVERITY_COLORS,
    put_text,
    put_text_right
)

ROW_HEIGHT = 27
LABEL_X = 16
VALUE_X = 120

MISSING = "--"

# At most this many alerts are listed at the bottom of the panel.
MAX_ALERTS = 3

# A weather multiplier at or above this reads as red rather than amber.
WEATHER_SEVERE = 1.30

# The key hints along the bottom of the panel. Every key the live loop
# in app/main.py handles belongs here: a control the driver cannot
# discover is a control they do not have. test_dashboard_renderer keeps
# the two in step.
KEY_HINTS = "Q quit   L landmarks   C calibrate   S sound   W weather"


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

        y = self._draw_safety_index(panel, y, state["risk"])

        self._draw_rows(panel, y, self._rows(state))

        self._draw_alerts(panel, state["risk"]["alerts"])

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

        # Naming the scoring mode on screen keeps the two engines apart
        # in a demo and in screenshots.
        mode = "DYNAMIC" if state["risk"]["dynamic"] else "STATIC"
        status = f"{mode}  {state['fps']:.1f} FPS" if SHOW_FPS else mode

        put_text_right(
            panel,
            status,
            self.panel_width - LABEL_X,
            34,
            0.5,
            MUTED
        )

        y = 50
        self._divider(panel, y)

        return y + 8

    def _draw_safety_index(self, panel, y, risk):

        right_x = self.panel_width - LABEL_X
        level = risk["level"]
        score = risk["score"]
        color = LEVEL_COLORS[level]

        put_text(panel, "SAFETY INDEX", (LABEL_X, y + 20), 0.45, MUTED)
        put_text_right(panel, level, right_x, y + 20, 0.55, color, 2)

        put_text(panel, f"{score:.0f}", (LABEL_X, y + 64), 1.3, color, 3)

        bar_left = LABEL_X + 95
        bar_top, bar_bottom = y + 46, y + 60

        cv2.rectangle(panel, (bar_left, bar_top), (right_x, bar_bottom),
                      DIVIDER, -1)

        filled = bar_left + int((right_x - bar_left) * score / 100)

        if filled > bar_left:
            cv2.rectangle(panel, (bar_left, bar_top), (filled, bar_bottom),
                          color, -1)

        y += 76

        # Why the score is being weighed the way it is, when it is being
        # weighed at all: a score that moves for invisible reasons is
        # worse than a blunt one.
        line = self._context_line(risk)

        if line:
            put_text(panel, line, (LABEL_X, y + 12), 0.42, MUTED)
            y += 20

        self._divider(panel, y)

        return y + 4

    @staticmethod
    def _context_line(risk):

        context = risk.get("context")

        if not risk.get("dynamic") or context is None:
            return None

        peak = max(context.fatigue_multiplier(),
                   context.attention_multiplier())

        if peak <= 1.01:
            return None

        summary = context.summary()

        return f"CONTEXT x{peak:.2f}" + (f"  {summary}" if summary else "")

    def _draw_alerts(self, panel, alerts):

        shown = alerts[:MAX_ALERTS]
        line_height = 22

        bottom = self.display_height - 44
        top = bottom - 26 - line_height * MAX_ALERTS

        self._divider(panel, top)

        put_text(panel, "ALERTS", (LABEL_X, top + 22), 0.45, MUTED)

        if len(alerts) > MAX_ALERTS:
            put_text_right(
                panel,
                f"+{len(alerts) - MAX_ALERTS} more",
                self.panel_width - LABEL_X,
                top + 22,
                0.45,
                MUTED
            )

        y = top + 22

        if not shown:
            put_text(panel, "None", (LABEL_X, y + line_height), 0.5, GOOD)
            return

        for alert in shown:
            y += line_height
            color = SEVERITY_COLORS[alert["severity"]]

            cv2.circle(panel, (LABEL_X + 5, y - 5), 5, color, -1)
            put_text(panel, alert["message"], (LABEL_X + 18, y), 0.5, color)

    def _rows(self, state):

        rows = []

        rows.extend(self._camera_rows(state))
        rows.extend(self._weather_rows(state))
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
        rows.extend(self._wheel_rows(state.get("wheel")))
        rows.append(None)
        rows.extend(self._phone_rows(state["phone"]))
        rows.extend(self._emotion_rows(state["emotion"]))
        rows.extend(self._seatbelt_rows(state["seatbelt"]))

        return rows

    @staticmethod
    def _camera_rows(state):
        """
        Only worth a row when the camera itself is the story. A working
        lens in a lit cabin is the assumption behind every other row, so
        saying so each frame would be noise.
        """

        camera = state.get("camera")

        if not camera:
            return []

        if camera["blocked"]:
            return [(
                "CAMERA",
                "BLOCKED",
                BAD,
                f"flat for {camera['blocked_duration']:.0f}s"
            )]

        strength = state.get("enhancement") or 0.0

        if strength > 0.0:
            return [(
                "CAMERA",
                "LOW LIGHT",
                WARN,
                f"lift {strength * 100:.0f}%, "
                f"light {camera['brightness']:.0f}"
            )]

        return []

    @staticmethod
    def _weather_rows(state):
        """
        Shown whenever the weather is weighing something, and whenever a
        forecast is involved even if it says clear.

        The second half is the point of the row. A driver who cannot see
        that the forecast is stale, or that it has not arrived, has no
        way of knowing there is anything to overrule - and the value
        in force says nothing about that on its own, because an honest
        1.0 and an ignorant one look identical.
        """

        context = state["risk"].get("context")

        if context is None:
            return []

        multiplier = context.weather_multiplier()
        source = context.weather_source

        if multiplier <= 1.0 and source == "default":
            return []

        if multiplier >= WEATHER_SEVERE:
            color = BAD
        elif multiplier > 1.0:
            color = WARN
        else:
            color = GOOD

        return [(
            "WEATHER",
            context.conditions.replace("_", " ").upper(),
            color,
            source
        )]

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

        # The threshold is this driver's own, so show what the EAR is
        # being measured against rather than the number alone.
        detail = f"EAR {blink['ear']:.2f}"

        if blink.get("threshold") is not None:
            detail += (
                f" / {blink['threshold']:.2f}"
                if blink.get("calibrated", True)
                else " calibrating"
            )

        return [
            (
                "EYES",
                "CLOSED" if closed else "OPEN",
                BAD if closed else GOOD,
                detail
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

    @staticmethod
    def _wheel_rows(wheel):
        """
        UNKNOWN is shown rather than hidden. On most setups the wheel
        sits below the camera's view, and a driver should be able to
        see that this is one of them instead of assuming the monitor
        is watching something it cannot see.
        """

        if not wheel:
            return [("HANDS", MISSING, MUTED, "")]

        status = wheel["status"]

        if status == "UNKNOWN":
            return [("HANDS", MISSING, MUTED, "wheel not in view")]

        value, color = {
            "BOTH": ("ON WHEEL", GOOD),
            "ONE": ("ONE HAND", TEXT),
            "NONE": ("OFF WHEEL", BAD),
        }[status]

        detail = (
            f"{wheel['missing_for']:.0f}s off"
            if status == "NONE"
            else f"{wheel['count']} seen"
        )

        return [("HANDS", value, color, detail)]

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

        put_text(panel, KEY_HINTS, (LABEL_X, y), 0.42, MUTED)

    def _divider(self, panel, y):

        cv2.line(
            panel,
            (LABEL_X, y),
            (self.panel_width - LABEL_X, y),
            DIVIDER,
            1
        )

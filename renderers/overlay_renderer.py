import math

import cv2

from renderers.theme import (
    GOOD,
    WARN,
    BAD,
    NEUTRAL_BOX,
    ACCENT,
    put_label,
    put_banner
)


class OverlayRenderer:
    """
    Draws detections on top of the camera frame: face boxes, irises, the
    gaze ray and status banners.
    """

    # (iris centre, one point on the iris edge)
    IRISES = [(468, 469), (473, 474)]

    # Between the eyes, where the gaze ray starts.
    NOSE_BRIDGE = 168

    def draw(self, frame, state):

        for face in state["faces"]:
            self._draw_face(frame, face)

        for phone in state["phone"]["phones"]:
            self._draw_phone(frame, phone, state["phone"]["in_use"])

        if state["landmarks"]:
            face_landmarks = state["landmarks"][0]
            self._draw_irises(frame, face_landmarks)
            self._draw_gaze(frame, face_landmarks, state)

        head_pose = state["head_pose"]

        if head_pose and not head_pose["calibrated"]:
            put_banner(
                frame,
                "CALIBRATING - LOOK STRAIGHT AHEAD",
                ACCENT,
                y=frame.shape[0] - 40
            )

        return frame

    def _draw_face(self, frame, face):

        x1, y1, x2, y2 = face["bbox"]

        color = GOOD if face["is_driver"] else NEUTRAL_BOX

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = f"{face['confidence']:.2f}"

        if face["id"] is not None:
            label = f"ID:{face['id']}  {label}"

        if face["is_driver"]:
            label = f"DRIVER  {label}"

        put_label(frame, label, x1, y1, color)

    def _draw_phone(self, frame, phone, in_use):

        x1, y1, x2, y2 = phone["bbox"]

        color = BAD if in_use else WARN

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        put_label(frame, f"PHONE  {phone['confidence']:.2f}", x1, y1,
                  color)

    def _draw_irises(self, frame, face):

        if len(face) <= max(max(pair) for pair in self.IRISES):
            return

        h, w = frame.shape[:2]

        for centre, edge in self.IRISES:
            cx, cy = face[centre].x * w, face[centre].y * h
            ex, ey = face[edge].x * w, face[edge].y * h

            radius = max(int(math.hypot(ex - cx, ey - cy)), 2)

            cv2.circle(frame, (int(cx), int(cy)), radius, ACCENT, 1,
                       cv2.LINE_AA)
            cv2.circle(frame, (int(cx), int(cy)), 2, ACCENT, -1)

    def _draw_gaze(self, frame, face, state):

        gaze = state["gaze"]
        head_pose = state["head_pose"]

        if not gaze or not head_pose:
            return

        h, w = frame.shape[:2]

        start = (face[self.NOSE_BRIDGE].x * w, face[self.NOSE_BRIDGE].y * h)

        # Scale the ray to the face: the distance between the outer eye
        # corners.
        eye_span = math.hypot(
            (face[263].x - face[33].x) * w,
            (face[263].y - face[33].y) * h
        )
        length = 1.5 * eye_span

        # Negative yaw (driver's left) points toward the image's right;
        # positive pitch (down) points down.
        dx = -math.sin(math.radians(gaze["gaze_yaw"])) * length
        dy = math.sin(math.radians(head_pose["pitch"])) * length

        distraction = state["distraction"]

        if distraction is None or distraction["eyes_on_road"]:
            color = GOOD
        elif distraction["distracted"]:
            color = BAD
        else:
            color = WARN

        cv2.arrowedLine(
            frame,
            (int(start[0]), int(start[1])),
            (int(start[0] + dx), int(start[1] + dy)),
            color,
            2,
            cv2.LINE_AA,
            tipLength=0.25
        )

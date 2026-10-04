import cv2

from renderers.theme import GOOD, NEUTRAL_BOX, ACCENT, put_label, put_banner


class OverlayRenderer:
    """
    Draws detections on top of the camera frame: face boxes and status
    banners.
    """

    def draw(self, frame, state):

        for face in state["faces"]:
            self._draw_face(frame, face)

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

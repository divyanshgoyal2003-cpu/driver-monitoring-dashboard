import cv2


class DashboardRenderer:

    def draw(self, frame, fps, face_count, blink_data):

        cv2.putText(
            frame,
            f"FPS : {fps:.2f}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Faces : {face_count}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2
        )

        if blink_data:

            cv2.putText(
                frame,
                f"EAR : {blink_data['ear']:.2f}",
                (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Blinks : {blink_data['blink_count']}",
                (20, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            status = "CLOSED" if blink_data["eyes_closed"] else "OPEN"

            color = (
                (0, 0, 255)
                if blink_data["eyes_closed"]
                else (0, 255, 0)
            )

            cv2.putText(
                frame,
                f"Eyes : {status}",
                (20, 175),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )

        return frame
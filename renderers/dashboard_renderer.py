import cv2


class DashboardRenderer:

    def draw(self, frame, fps, face_count, blink_data, yawn_data, head_pose_data):

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
        if yawn_data:

            cv2.putText(
                frame,
                f"MAR : {yawn_data['mar']:.2f}",
                (20, 210),
                cv2.FONT_HERSHEY_SIMPLEX,
                 0.8,
                (255, 165, 0),
                2
             )

            cv2.putText(
                frame,
                f"Yawns : {yawn_data['yawn_count']}",
                (20, 245),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                 2
            )

            mouth_status = "OPEN" if yawn_data["mouth_open"] else "CLOSED"

            cv2.putText(
                frame,
                f"Mouth : {mouth_status}",
                (20, 280),
                cv2.FONT_HERSHEY_SIMPLEX,
                 0.8,
                (0, 255, 0),
                 2
            )
        if head_pose_data:

            cv2.putText(
                frame,
                f"Yaw : {head_pose_data['yaw']:.1f}",
                (20, 315),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Pitch : {head_pose_data['pitch']:.1f}",
                (20, 345),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Roll : {head_pose_data['roll']:.1f}",
                (20, 375),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Direction : {head_pose_data['direction']}",
                (20, 405),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )
        return frame
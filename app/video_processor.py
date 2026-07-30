import cv2
import time

from models.detectors.yolo_face_detector import YOLOFaceDetector
from app.config import FACE_LANDMARK_MODEL
from models.landmarks.mediapipe_face_landmarker import MediaPipeFaceLandmarker


class VideoProcessor:
    def __init__(self):
        self.prev_time = time.time()
        self.face_detector = YOLOFaceDetector()
        self.landmarker = MediaPipeFaceLandmarker(FACE_LANDMARK_MODEL)

    def process(self, frame):

        current_time = time.time()

        fps = 1 / (current_time - self.prev_time)

        self.prev_time = current_time

        frame, faces = self.face_detector.track(frame)
        result = self.landmarker.detect(frame)
        # if result.face_landmarks:
        #     for face_landmarks in result.face_landmarks:

        #         vision.drawing_utils.draw_landmarks(
        #             image=frame,
        #             landmark_list=face_landmarks,
        #             connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
        #             landmark_drawing_spec=None,
        #             connection_drawing_spec=vision.drawing_styles.get_default_face_mesh_tesselation_style()
        #         )
        if result.face_landmarks:
            h, w, _ = frame.shape

            for face_landmarks in result.face_landmarks:
                for landmark in face_landmarks:
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)

                    cv2.circle(
                        frame,
                        (x, y),
                        1,
                        (0, 255, 255),
                        -1
                    )

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
            f"Faces : {len(faces)}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2
        )

        return frame
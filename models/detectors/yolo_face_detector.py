from ultralytics import YOLO
import cv2


class YOLOFaceDetector:
    def __init__(self, model_path="trained_models/face/face_yolov8n.pt"):
        self.model = YOLO(model_path)

    def detect(self, frame, confidence=0.4):
        """
        Detect faces in a frame.

        Returns:
            annotated_frame
            faces -> list of dictionaries
        """

        results = self.model.predict(
            source=frame,
            conf=confidence,
            verbose=False
        )

        annotated_frame = frame.copy()
        faces = []

        for result in results:
            for box in result.boxes:

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                conf = float(box.conf[0])

                faces.append({
                    "bbox": (x1, y1, x2, y2),
                    "confidence": conf
                })

                cv2.rectangle(
                    annotated_frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    annotated_frame,
                    f"{conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )

        return annotated_frame, faces
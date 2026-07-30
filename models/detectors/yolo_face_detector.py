from ultralytics import YOLO
import cv2


class YOLOFaceDetector:

    def __init__(self,
                 model_path="trained_models/face/face_yolov8n.pt"):

        self.model = YOLO(model_path)

    def detect(self, frame, confidence=0.4):

        results = self.model.predict(
            source=frame,
            conf=confidence,
            verbose=False,
            device="cpu"
        )

        return self._draw_results(frame, results)

    def track(self, frame, confidence=0.4):

        results = self.model.track(
            source=frame,
            conf=confidence,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False,
            device="cpu"
        )

        return self._draw_results(frame, results)

    def _draw_results(self, frame, results):

        annotated = frame.copy()
        faces = []

        for result in results:

            if result.boxes is None:
                continue

            for box in result.boxes:

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                conf = float(box.conf[0])

                track_id = None

                if box.id is not None:
                    track_id = int(box.id[0])

                faces.append({
                    "id": track_id,
                    "bbox": (x1, y1, x2, y2),
                    "confidence": conf
                })

                cv2.rectangle(
                    annotated,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                label = (
                    f"ID:{track_id}  {conf:.2f}"
                    if track_id is not None
                    else f"{conf:.2f}"
                )

                cv2.putText(
                    annotated,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

        return annotated, faces
from ultralytics import YOLO

from app.config import (
    YOLO_FACE_MODEL,
    FACE_DETECTION_CONFIDENCE,
    FACE_DETECTION_IMAGE_SIZE
)


class YOLOFaceDetector:
    """
    Detects and tracks faces. Returns plain face records; drawing them is
    left to renderers.overlay_renderer.
    """

    def __init__(self, model_path=YOLO_FACE_MODEL,
                 confidence=FACE_DETECTION_CONFIDENCE,
                 image_size=FACE_DETECTION_IMAGE_SIZE):

        self.model = YOLO(model_path)
        self.confidence = confidence
        self.image_size = image_size

    def detect(self, frame):

        results = self.model.predict(
            source=frame,
            conf=self.confidence,
            imgsz=self.image_size,
            verbose=False,
            device="cpu"
        )

        return self._parse_results(results)

    def track(self, frame):

        results = self.model.track(
            source=frame,
            conf=self.confidence,
            imgsz=self.image_size,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False,
            device="cpu"
        )

        return self._parse_results(results)

    def _parse_results(self, results):

        faces = []

        for result in results:

            if result.boxes is None:
                continue

            for box in result.boxes:

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                track_id = None

                if box.id is not None:
                    track_id = int(box.id[0])

                faces.append({
                    "id": track_id,
                    "bbox": (x1, y1, x2, y2),
                    "confidence": float(box.conf[0]),
                    "is_driver": False
                })

        return faces

from ultralytics import YOLO

from app.config import OBJECT_DETECTION_MODEL, PHONE_CONFIDENCE


class YOLOObjectDetector:
    """
    Detects selected object classes with a COCO-trained YOLO model.
    By default it looks for phones only.
    """

    def __init__(self, model_path=OBJECT_DETECTION_MODEL,
                 class_names=("cell phone",),
                 confidence=PHONE_CONFIDENCE):

        self.model = YOLO(model_path)
        self.confidence = confidence

        # Look the class IDs up by name rather than hard-coding COCO's
        # numbering, and fail loudly if the model lacks a class.
        ids_by_name = {name: i for i, name in self.model.names.items()}

        missing = [name for name in class_names if name not in ids_by_name]

        if missing:
            raise ValueError(
                f"{model_path} has no class named {missing}"
            )

        self.class_ids = [ids_by_name[name] for name in class_names]

    def detect(self, frame):

        results = self.model.predict(
            source=frame,
            conf=self.confidence,
            classes=self.class_ids,
            verbose=False,
            device="cpu"
        )

        detections = []

        for result in results:

            if result.boxes is None:
                continue

            for box in result.boxes:

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                detections.append({
                    "label": self.model.names[int(box.cls[0])],
                    "bbox": (x1, y1, x2, y2),
                    "confidence": float(box.conf[0])
                })

        return detections

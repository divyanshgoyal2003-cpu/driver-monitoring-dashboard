import cv2
import numpy as np

from app.config import EMOTION_MODEL, EMOTION_FACE_MARGIN


class EmotionClassifier:
    """
    Facial expression recognition with FER+ (Microsoft's FER2013 dataset
    relabelled by 10 annotators per image), from the ONNX model zoo.

    Input is a 64x64 grayscale face with raw 0-255 pixel values; output
    is one score per emotion, turned into probabilities with a softmax.
    """

    # Output order of the FER+ model
    LABELS = (
        "Neutral",
        "Happy",
        "Surprised",
        "Sad",
        "Angry",
        "Disgusted",
        "Fearful",
        "Contempt"
    )

    INPUT_SIZE = 64
    MIN_FACE_SIZE = 24

    def __init__(self, model_path=EMOTION_MODEL, margin=EMOTION_FACE_MARGIN):

        self.net = cv2.dnn.readNetFromONNX(model_path)
        self.margin = margin

    def predict(self, frame, box):
        """
        Probabilities in LABELS order for the face inside `box`
        (x1, y1, x2, y2), or None if the face is too small to classify.
        """

        crop = self._crop(frame, box)

        if crop is None:
            return None

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        gray = cv2.resize(
            gray,
            (self.INPUT_SIZE, self.INPUT_SIZE),
            interpolation=cv2.INTER_AREA
        )

        blob = gray.astype(np.float32).reshape(
            1, 1, self.INPUT_SIZE, self.INPUT_SIZE
        )

        self.net.setInput(blob)
        scores = self.net.forward()[0].astype(np.float64)

        exp = np.exp(scores - scores.max())

        return exp / exp.sum()

    def _crop(self, frame, box):

        frame_h, frame_w = frame.shape[:2]
        x1, y1, x2, y2 = box

        half = max(x2 - x1, y2 - y1) * self.margin / 2
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

        left = int(max(cx - half, 0))
        top = int(max(cy - half, 0))
        right = int(min(cx + half, frame_w))
        bottom = int(min(cy + half, frame_h))

        if min(right - left, bottom - top) < self.MIN_FACE_SIZE:
            return None

        return frame[top:bottom, left:right]

"""
Finding the driver's hands.

A thin wrapper over MediaPipe's hand landmarker, reporting only what
the wheel monitor needs: how many hands are in the frame and how
confident the detector is about each.

VIDEO mode, like the face landmarker, so hands are tracked between
frames rather than detected from scratch each time. It needs a
strictly increasing timestamp.
"""

import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from app.config import HAND_LANDMARK_MODEL, HAND_CONFIDENCE


class MediaPipeHandLandmarker:

    def __init__(self, model_path=HAND_LANDMARK_MODEL, num_hands=2,
                 confidence=HAND_CONFIDENCE):

        self.confidence = confidence

        base_options = python.BaseOptions(model_asset_path=model_path)

        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=confidence,
            min_hand_presence_confidence=confidence,
            min_tracking_confidence=confidence
        )

        self.landmarker = vision.HandLandmarker.create_from_options(options)

        self.last_timestamp_ms = -1

    def detect(self, frame, timestamp):
        """
        {"count": int, "confidence": float, "hands": [...]}.

        confidence is the best score among the hands found, or 0.0 when
        there are none.
        """

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        timestamp_ms = max(
            int(timestamp * 1000), self.last_timestamp_ms + 1
        )
        self.last_timestamp_ms = timestamp_ms

        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

        scores = [
            hand[0].score
            for hand in (result.handedness or [])
            if hand
        ]

        return {
            "count": len(result.hand_landmarks or []),
            "confidence": max(scores) if scores else 0.0,
            "hands": result.hand_landmarks or []
        }

    def close(self):
        self.landmarker.close()

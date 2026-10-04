import numpy as np

from app.config import (
    EMOTION_SMOOTHING,
    NEGATIVE_EMOTIONS,
    EMOTION_ALERT_CONFIDENCE
)


class EmotionAnalyzer:
    """
    Smooths per-frame emotion probabilities with an exponential moving
    average, so the displayed emotion does not flicker between frames,
    and flags emotions that can impair driving.
    """

    def __init__(self, labels, smoothing=EMOTION_SMOOTHING,
                 negative_emotions=NEGATIVE_EMOTIONS,
                 alert_confidence=EMOTION_ALERT_CONFIDENCE):

        self.labels = list(labels)
        self.smoothing = smoothing
        self.negative_emotions = set(negative_emotions)
        self.alert_confidence = alert_confidence

        self.smoothed = None

    def update(self, probabilities, face_present):
        """
        probabilities: this frame's prediction, or None when the
        classifier did not run on this frame (the last average is kept).
        """

        if not face_present:
            # Do not carry one face's expression over to the next.
            self.smoothed = None
            return None

        if probabilities is not None:
            probabilities = np.asarray(probabilities, dtype=np.float64)

            if self.smoothed is None:
                self.smoothed = probabilities
            else:
                self.smoothed = (
                    self.smoothing * probabilities
                    + (1 - self.smoothing) * self.smoothed
                )

        if self.smoothed is None:
            return None

        best = int(np.argmax(self.smoothed))
        emotion = self.labels[best]
        confidence = float(self.smoothed[best])

        return {
            "emotion": emotion,
            "confidence": confidence,
            "probabilities": dict(zip(self.labels,
                                      map(float, self.smoothed))),
            "negative": (
                emotion in self.negative_emotions
                and confidence >= self.alert_confidence
            )
        }

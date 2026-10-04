import unittest

from analytics.emotion_analyzer import EmotionAnalyzer

LABELS = ("Neutral", "Happy", "Angry")


class EmotionAnalyzerTest(unittest.TestCase):

    def setUp(self):
        self.analyzer = EmotionAnalyzer(
            LABELS,
            smoothing=0.5,
            negative_emotions=("Angry",),
            alert_confidence=0.5
        )

    def test_first_prediction_is_used_directly(self):
        result = self.analyzer.update([0.1, 0.8, 0.1], True)
        self.assertEqual(result["emotion"], "Happy")
        self.assertAlmostEqual(result["confidence"], 0.8)
        self.assertFalse(result["negative"])

    def test_predictions_are_averaged(self):
        self.analyzer.update([1.0, 0.0, 0.0], True)
        result = self.analyzer.update([0.0, 0.0, 1.0], True)
        # 0.5 * new + 0.5 * old
        self.assertAlmostEqual(result["probabilities"]["Neutral"], 0.5)
        self.assertAlmostEqual(result["probabilities"]["Angry"], 0.5)

    def test_single_frame_spike_does_not_flip_the_emotion(self):
        analyzer = EmotionAnalyzer(LABELS, smoothing=0.3)

        for _ in range(5):
            analyzer.update([0.9, 0.05, 0.05], True)

        result = analyzer.update([0.0, 0.0, 1.0], True)
        self.assertEqual(result["emotion"], "Neutral")

    def test_frames_without_prediction_keep_the_average(self):
        self.analyzer.update([0.1, 0.8, 0.1], True)
        result = self.analyzer.update(None, True)
        self.assertEqual(result["emotion"], "Happy")

    def test_no_prediction_yet(self):
        self.assertIsNone(self.analyzer.update(None, True))

    def test_face_lost_resets(self):
        self.analyzer.update([0.1, 0.8, 0.1], True)
        self.assertIsNone(self.analyzer.update(None, False))
        self.assertIsNone(self.analyzer.update(None, True))

    def test_negative_emotion_needs_confidence(self):
        weak = self.analyzer.update([0.3, 0.3, 0.4], True)
        self.assertEqual(weak["emotion"], "Angry")
        self.assertFalse(weak["negative"])

        analyzer = EmotionAnalyzer(LABELS, negative_emotions=("Angry",),
                                   alert_confidence=0.5)
        strong = analyzer.update([0.1, 0.1, 0.8], True)
        self.assertTrue(strong["negative"])


if __name__ == "__main__":
    unittest.main()

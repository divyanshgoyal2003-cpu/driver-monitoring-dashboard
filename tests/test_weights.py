import unittest

from risk_engine import weights
from app.config import (
    SEVERITY_FLOOR,
    CONFIDENCE_FLOOR,
    CONFIDENCE_SEATBELT_AT_BOUNDARY,
    SEATBELT_OFF_RATIO,
    EYE_CONDITIONS
)

ALL_CONDITIONS = (
    "microsleep", "phone", "distracted", "driver_absent", "drowsy",
    "tired", "no_seatbelt", "frequent_yawning", "negative_emotion"
)


class SeverityTest(unittest.TestCase):

    def test_missing_measurements_weigh_full(self):
        """
        A detector returning partial data must not quietly soften the
        score.
        """

        for condition in ALL_CONDITIONS:
            self.assertEqual(weights.severity(condition, {}), 1.0)

    def test_severity_stays_within_its_band(self):
        states = [
            {"distraction": {"off_road_duration": d}} for d in
            (0, 1, 2, 3, 6, 20, 1000)
        ]

        for state in states:
            value = weights.severity("distracted", state)
            self.assertGreaterEqual(value, SEVERITY_FLOOR)
            self.assertLessEqual(value, 1.0)

    def test_a_longer_glance_away_weighs_more(self):
        short = weights.severity(
            "distracted", {"distraction": {"off_road_duration": 2.0}}
        )
        medium = weights.severity(
            "distracted", {"distraction": {"off_road_duration": 4.0}}
        )
        long = weights.severity(
            "distracted", {"distraction": {"off_road_duration": 8.0}}
        )

        self.assertAlmostEqual(short, SEVERITY_FLOOR)
        self.assertLess(short, medium)
        self.assertLess(medium, long)
        self.assertAlmostEqual(long, 1.0)

    def test_a_longer_eye_closure_weighs_more(self):
        brief = weights.severity(
            "microsleep", {"blink": {"closed_duration": 1.5}}
        )
        long = weights.severity(
            "microsleep", {"blink": {"closed_duration": 4.0}}
        )

        self.assertLess(brief, long)
        self.assertAlmostEqual(long, 1.0)

    def test_deeper_perclos_weighs_more(self):
        light = weights.severity("drowsy", {"drowsiness": {"perclos": 0.25}})
        heavy = weights.severity("drowsy", {"drowsiness": {"perclos": 0.45}})

        self.assertLess(light, heavy)

    def test_more_yawns_weigh_more(self):
        three = weights.severity("frequent_yawning",
                                 {"yawn": {"recent_yawns": 3}})
        eight = weights.severity("frequent_yawning",
                                 {"yawn": {"recent_yawns": 8}})

        self.assertLess(three, eight)

    def test_a_longer_absence_weighs_more(self):
        brief = weights.severity("driver_absent", {}, absent_duration=2.0)
        long = weights.severity("driver_absent", {}, absent_duration=8.0)

        self.assertLess(brief, long)

    def test_conditions_without_a_magnitude_weigh_full(self):
        for condition in ("no_seatbelt", "negative_emotion"):
            self.assertEqual(
                weights.severity(condition, {"seatbelt": {"ratio": 0.1}}),
                1.0
            )


class ConfidenceTest(unittest.TestCase):

    def test_missing_measurements_are_trusted(self):
        for condition in ALL_CONDITIONS:
            self.assertEqual(weights.confidence(condition, {}), 1.0)

    def test_confidence_stays_within_its_band(self):
        for ratio in (0.0, 0.1, 0.2, 0.5, 1.0):
            value = weights.confidence(
                "no_seatbelt", {"seatbelt": {"ratio": ratio}}
            )
            self.assertGreaterEqual(value, CONFIDENCE_FLOOR)
            self.assertLessEqual(value, 1.0)

    def test_a_borderline_seatbelt_verdict_is_trusted_least(self):
        certain = weights.confidence(
            "no_seatbelt", {"seatbelt": {"ratio": 0.0}}
        )
        boundary = weights.confidence(
            "no_seatbelt", {"seatbelt": {"ratio": SEATBELT_OFF_RATIO}}
        )

        self.assertAlmostEqual(certain, 1.0)
        self.assertAlmostEqual(boundary, CONFIDENCE_SEATBELT_AT_BOUNDARY)
        self.assertLess(boundary, certain)

    def test_a_weak_emotion_reading_is_trusted_less(self):
        weak = weights.confidence(
            "negative_emotion", {"emotion": {"confidence": 0.5}}
        )
        strong = weights.confidence(
            "negative_emotion", {"emotion": {"confidence": 0.95}}
        )

        self.assertLess(weak, strong)
        self.assertAlmostEqual(strong, 1.0)

    def test_a_low_confidence_phone_box_is_trusted_less(self):
        weak = weights.confidence(
            "phone", {"phone": {"phones": [{"confidence": 0.35}]}}
        )
        strong = weights.confidence(
            "phone", {"phone": {"phones": [{"confidence": 0.9}]}}
        )

        self.assertLess(weak, strong)
        self.assertAlmostEqual(strong, 1.0)

    def test_the_best_box_decides(self):
        value = weights.confidence(
            "phone",
            {"phone": {"phones": [{"confidence": 0.36},
                                  {"confidence": 0.95}]}}
        )

        self.assertAlmostEqual(value, 1.0)

    def test_eye_conditions_are_trusted_less_in_the_dark(self):
        for condition in EYE_CONDITIONS:
            lit = weights.confidence(condition, {}, darkness=0.0)
            dark = weights.confidence(condition, {}, darkness=1.0)

            self.assertAlmostEqual(lit, 1.0)
            self.assertLess(dark, lit)

    def test_darkness_does_not_touch_other_detectors(self):
        for condition in ("phone", "no_seatbelt", "negative_emotion"):
            self.assertEqual(
                weights.confidence(condition, {}, darkness=1.0), 1.0
            )

    def test_a_poor_calibration_lowers_eye_confidence(self):
        good = weights.confidence(
            "drowsy", {"blink": {"eye_confidence": 1.0}}
        )
        poor = weights.confidence(
            "drowsy", {"blink": {"eye_confidence": 0.5}}
        )

        self.assertLess(poor, good)

    def test_poor_calibration_and_darkness_compound(self):
        both = weights.confidence(
            "drowsy", {"blink": {"eye_confidence": 0.5}}, darkness=1.0
        )
        either = weights.confidence(
            "drowsy", {"blink": {"eye_confidence": 0.5}}, darkness=0.0
        )

        self.assertLess(both, either)
        self.assertGreaterEqual(both, CONFIDENCE_FLOOR)


if __name__ == "__main__":
    unittest.main()

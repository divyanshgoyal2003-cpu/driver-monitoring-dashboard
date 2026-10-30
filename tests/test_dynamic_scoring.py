"""
The three weighting layers, seen through the Driver Safety Index.

These assert relationships ("a longer glance costs more than a shorter
one") rather than magic constants, so retuning a ramp in config.py does
not rewrite the test suite.
"""

import unittest

from risk_engine.scoring import DriverSafetyIndex
from risk_engine.context import DrivingContext
from app.config import RISK_PENALTIES


def make_state(**changes):
    """A frame where everything is fine, with the given changes."""

    state = {
        "face_detected": True,
        "blink": {"closed_duration": 0.0, "eye_confidence": 1.0},
        "drowsiness": {"state": "AWAKE", "perclos": 0.0},
        "yawn": {"yawn_count": 0, "recent_yawns": 0},
        "distraction": {"distracted": False, "off_road_duration": 0.0},
        "phone": {"in_use": False, "usage_duration": 0.0, "phones": []},
        "seatbelt": {"status": "ON", "ratio": 1.0},
        "emotion": {"negative": False, "confidence": 0.0},
    }

    state.update(changes)

    return state


def distracted(duration):
    return make_state(
        distraction={"distracted": True, "off_road_duration": duration}
    )


def score_once(state, context=None, dynamic=True):
    """Penalties from a single frame, before any smoothing applies."""

    index = DriverSafetyIndex(dynamic=dynamic)

    return index.update(state, 0.0, context)


class SeverityScalingTest(unittest.TestCase):

    def test_a_longer_distraction_costs_more(self):
        brief = score_once(distracted(2.0))["penalties"]["distracted"]
        long = score_once(distracted(8.0))["penalties"]["distracted"]

        self.assertLess(brief, long)
        self.assertAlmostEqual(long, RISK_PENALTIES["distracted"])

    def test_a_brief_distraction_can_stay_inside_SAFE(self):
        """
        The flat table charged a two second glance the same 30 points as
        a twenty second stare, which was enough to leave SAFE on its own.
        """

        brief = score_once(distracted(2.0))
        long = score_once(distracted(10.0))

        self.assertEqual(brief["level"], "SAFE")
        self.assertEqual(long["level"], "CAUTION")

    def test_a_longer_eye_closure_costs_more(self):
        brief = score_once(make_state(
            drowsiness={"state": "MICROSLEEP", "perclos": 0.3},
            blink={"closed_duration": 1.5, "eye_confidence": 1.0},
        ))["penalties"]["microsleep"]

        long = score_once(make_state(
            drowsiness={"state": "MICROSLEEP", "perclos": 0.3},
            blink={"closed_duration": 4.0, "eye_confidence": 1.0},
        ))["penalties"]["microsleep"]

        self.assertLess(brief, long)


class ConfidenceScalingTest(unittest.TestCase):

    def test_a_borderline_seatbelt_costs_less_than_a_certain_one(self):
        boundary = score_once(
            make_state(seatbelt={"status": "OFF", "ratio": 0.2})
        )["penalties"]["no_seatbelt"]

        certain = score_once(
            make_state(seatbelt={"status": "OFF", "ratio": 0.0})
        )["penalties"]["no_seatbelt"]

        self.assertLess(boundary, certain)
        self.assertAlmostEqual(certain, RISK_PENALTIES["no_seatbelt"])

    def test_a_weak_phone_detection_costs_less(self):
        weak = score_once(make_state(phone={
            "in_use": True, "usage_duration": 3.0,
            "phones": [{"confidence": 0.35}],
        }))["penalties"]["phone"]

        strong = score_once(make_state(phone={
            "in_use": True, "usage_duration": 3.0,
            "phones": [{"confidence": 0.95}],
        }))["penalties"]["phone"]

        self.assertLess(weak, strong)

    def test_the_recorded_false_positive_no_longer_leaves_SAFE(self):
        """
        The session that prompted this work: a borderline seatbelt
        verdict and a low-confidence phone box together took the score
        to CAUTION while the driver sat still at a desk.
        """

        state = make_state(
            seatbelt={"status": "OFF", "ratio": 0.19},
            phone={"in_use": True, "usage_duration": 0.6,
                   "phones": [{"confidence": 0.36}]},
        )

        self.assertEqual(score_once(state, dynamic=False)["level"],
                         "CAUTION")
        self.assertEqual(score_once(state, dynamic=True)["level"], "SAFE")


class ContextScalingTest(unittest.TestCase):

    def drowsy_at(self, hour):
        state = make_state(
            drowsiness={"state": "DROWSY", "perclos": 0.45},
            blink={"closed_duration": 0.0, "eye_confidence": 1.0},
        )

        return score_once(state, DrivingContext(hour=hour, brightness=150))

    def test_the_small_hours_cost_more_than_midday(self):
        night = self.drowsy_at(3)["penalties"]["drowsy"]
        noon = self.drowsy_at(12)["penalties"]["drowsy"]

        self.assertGreater(night, noon)

    def test_time_at_the_wheel_costs_more(self):
        state = make_state(
            drowsiness={"state": "TIRED", "perclos": 0.2},
            blink={"closed_duration": 0.0, "eye_confidence": 1.0},
        )

        fresh = score_once(
            state, DrivingContext(hour=12, elapsed=0, brightness=150)
        )["penalties"]["tired"]

        late = score_once(
            state, DrivingContext(hour=12, elapsed=3 * 3600,
                                  brightness=150)
        )["penalties"]["tired"]

        self.assertGreater(late, fresh)

    def test_context_leaves_the_seatbelt_alone(self):
        state = make_state(seatbelt={"status": "OFF", "ratio": 0.0})

        night = score_once(state, DrivingContext(hour=3))
        noon = score_once(state, DrivingContext(hour=12))

        self.assertAlmostEqual(night["penalties"]["no_seatbelt"],
                               noon["penalties"]["no_seatbelt"])

    def test_darkness_raises_attention_but_lowers_eye_confidence(self):
        """
        The counterweight: a dark cabin makes a glance away matter more,
        while making the eye measurements worth less. Without the second
        half the engine would amplify its own noise at night.
        """

        dark = DrivingContext(hour=3, brightness=10)
        lit = DrivingContext(hour=3, brightness=200)

        glance = distracted(8.0)
        self.assertGreater(
            score_once(glance, dark)["penalties"]["distracted"],
            score_once(glance, lit)["penalties"]["distracted"],
        )

        drowsy = make_state(
            drowsiness={"state": "DROWSY", "perclos": 0.45},
            blink={"closed_duration": 0.0, "eye_confidence": 1.0},
        )
        self.assertLess(
            score_once(drowsy, dark)["penalties"]["drowsy"],
            score_once(drowsy, lit)["penalties"]["drowsy"],
        )

    def test_an_uncertain_night_reading_is_no_worse_than_a_confident_day_one(self):
        drowsy = make_state(
            drowsiness={"state": "DROWSY", "perclos": 0.45},
            blink={"closed_duration": 0.0, "eye_confidence": 1.0},
        )

        night = score_once(
            drowsy, DrivingContext(hour=3, brightness=10)
        )["penalties"]["drowsy"]

        day = score_once(
            drowsy, DrivingContext(hour=12, brightness=200)
        )["penalties"]["drowsy"]

        self.assertLessEqual(night, day * 1.05)


class InvariantTest(unittest.TestCase):

    def test_severity_and_confidence_only_ever_reduce(self):
        """
        With no context, nothing can cost more than its entry in
        RISK_PENALTIES.
        """

        states = [
            distracted(100.0),
            make_state(phone={"in_use": True, "usage_duration": 100.0,
                              "phones": [{"confidence": 1.0}]}),
            make_state(seatbelt={"status": "OFF", "ratio": 0.0}),
            make_state(drowsiness={"state": "MICROSLEEP", "perclos": 1.0},
                       blink={"closed_duration": 100.0,
                              "eye_confidence": 1.0}),
            make_state(yawn={"yawn_count": 50, "recent_yawns": 50}),
            make_state(emotion={"negative": True, "confidence": 1.0}),
        ]

        for state in states:
            for name, penalty in score_once(state)["penalties"].items():
                self.assertLessEqual(penalty, RISK_PENALTIES[name] + 1e-9,
                                     f"{name} exceeded its base")

    def test_alerts_do_not_soften_with_confidence(self):
        """
        Confidence scales what a risk costs, never whether the driver is
        warned about it.
        """

        result = score_once(
            make_state(seatbelt={"status": "OFF", "ratio": 0.2})
        )

        self.assertLess(result["penalties"]["no_seatbelt"],
                        RISK_PENALTIES["no_seatbelt"])
        self.assertEqual(result["alerts"][0]["message"],
                         "SEATBELT NOT DETECTED")

    def test_static_mode_ignores_every_layer(self):
        state = distracted(2.0)
        context = DrivingContext(hour=3, elapsed=4 * 3600, brightness=10)

        result = score_once(state, context, dynamic=False)

        self.assertEqual(result["penalties"],
                         {"distracted": RISK_PENALTIES["distracted"]})
        self.assertEqual(result["multipliers"], {})
        self.assertFalse(result["dynamic"])

    def test_the_three_factors_are_reported(self):
        result = score_once(distracted(3.0), DrivingContext(hour=3))
        factors = result["multipliers"]["distracted"]

        self.assertEqual(set(factors), {"severity", "confidence", "context"})
        self.assertTrue(result["dynamic"])

    def test_an_empty_state_scores_like_the_flat_table(self):
        """
        The old minimal state dicts must keep working: anything the
        engine cannot measure weighs 1.0.
        """

        bare = {
            "face_detected": True,
            "drowsiness": {"state": "AWAKE"},
            "yawn": {"yawn_count": 0, "recent_yawns": 0},
            "distraction": {"distracted": True},
            "phone": {"in_use": False},
            "seatbelt": {"status": "ON"},
            "emotion": {"negative": False},
        }

        self.assertAlmostEqual(
            score_once(bare)["penalties"]["distracted"],
            RISK_PENALTIES["distracted"],
        )


if __name__ == "__main__":
    unittest.main()

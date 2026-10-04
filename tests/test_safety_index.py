import unittest

from risk_engine.scoring import DriverSafetyIndex


def make_state(**changes):
    """A frame where everything is fine, with the given changes."""

    state = {
        "face_detected": True,
        "drowsiness": {"state": "AWAKE"},
        "yawn": {"yawn_count": 0, "recent_yawns": 0},
        "distraction": {"distracted": False},
        "phone": {"in_use": False},
        "seatbelt": {"status": "ON"},
        "emotion": {"negative": False},
    }

    state.update(changes)

    return state


def hold(index, state, start, seconds, fps=30.0):
    """Feed the same state for `seconds`. Returns (results, next t)."""

    results = []
    t = start

    for _ in range(int(round(seconds * fps))):
        results.append(index.update(state, t))
        t += 1 / fps

    return results, t


class DriverSafetyIndexTest(unittest.TestCase):

    def setUp(self):
        self.index = DriverSafetyIndex()

    def test_all_clear(self):
        results, _ = hold(self.index, make_state(), 0.0, 1.0)
        result = results[-1]
        self.assertAlmostEqual(result["score"], 100.0)
        self.assertEqual(result["level"], "SAFE")
        self.assertEqual(result["alerts"], [])
        self.assertTrue(all(not r["events"] for r in results))

    def test_phone_use_lowers_the_score(self):
        _, t = hold(self.index, make_state(), 0.0, 1.0)
        results, _ = hold(self.index, make_state(phone={"in_use": True}),
                          t, 3.0)
        result = results[-1]

        self.assertAlmostEqual(result["raw_score"], 65.0)
        self.assertAlmostEqual(result["score"], 65.0, delta=0.5)
        self.assertEqual(result["level"], "CAUTION")
        self.assertEqual(result["alerts"][0]["message"], "PHONE IN USE")
        self.assertEqual(result["penalties"], {"phone": 35})

    def test_microsleep_is_danger_within_a_second(self):
        _, t = hold(self.index, make_state(), 0.0, 1.0)
        results, _ = hold(
            self.index, make_state(drowsiness={"state": "MICROSLEEP"}),
            t, 1.0
        )
        self.assertEqual(results[-1]["level"], "DANGER")
        self.assertEqual(results[-1]["alerts"][0]["severity"], "critical")

    def test_score_never_goes_below_zero(self):
        everything = make_state(
            drowsiness={"state": "MICROSLEEP"},
            phone={"in_use": True},
            distraction={"distracted": True},
            seatbelt={"status": "OFF"},
        )
        results, _ = hold(self.index, everything, 0.0, 2.0)
        self.assertEqual(results[-1]["raw_score"], 0.0)
        self.assertGreaterEqual(results[-1]["score"], 0.0)

    def test_recovery_is_slower_than_the_fall(self):
        _, t = hold(self.index, make_state(), 0.0, 1.0)
        _, t = hold(self.index, make_state(phone={"in_use": True}), t, 3.0)
        results, _ = hold(self.index, make_state(), t, 1.0)

        # One second after the phone is put away the score is still
        # recovering (rise time constant 5 s).
        self.assertLess(results[-1]["score"], 75.0)
        self.assertGreater(results[-1]["score"], 65.0)
        self.assertEqual(results[-1]["alerts"], [])

    def test_driver_absent_after_two_seconds(self):
        _, t = hold(self.index, make_state(), 0.0, 1.0)

        no_face = make_state(face_detected=False, drowsiness=None,
                             yawn=None, distraction=None, emotion=None)

        results, t = hold(self.index, no_face, t, 1.5)
        self.assertEqual(results[-1]["alerts"], [])

        results, _ = hold(self.index, no_face, t, 1.0)
        self.assertEqual(results[-1]["alerts"][0]["message"],
                         "DRIVER NOT VISIBLE")

    def test_events_fire_once_per_episode(self):
        _, t = hold(self.index, make_state(), 0.0, 1.0)
        results, t = hold(self.index, make_state(phone={"in_use": True}),
                          t, 2.0)

        phone_events = [
            event for r in results for event in r["events"]
            if event["type"] == "PHONE"
        ]
        self.assertEqual(len(phone_events), 1)
        self.assertEqual(phone_events[0]["severity"], "high")

    def test_yawn_event(self):
        _, t = hold(self.index, make_state(), 0.0, 1.0)

        result = self.index.update(
            make_state(yawn={"yawn_count": 1, "recent_yawns": 1}), t
        )

        self.assertEqual([e["type"] for e in result["events"]], ["YAWN"])

        result = self.index.update(
            make_state(yawn={"yawn_count": 1, "recent_yawns": 1}), t + 0.1
        )
        self.assertEqual(result["events"], [])

    def test_high_risk_event_on_entering_danger(self):
        _, t = hold(self.index, make_state(), 0.0, 1.0)
        results, _ = hold(
            self.index, make_state(drowsiness={"state": "MICROSLEEP"}),
            t, 2.0
        )

        types = [e["type"] for r in results for e in r["events"]]
        self.assertEqual(types.count("HIGH_RISK"), 1)
        self.assertEqual(types.count("MICROSLEEP"), 1)

    def test_alerts_are_ordered_by_urgency(self):
        state = make_state(
            seatbelt={"status": "OFF"},
            drowsiness={"state": "MICROSLEEP"},
            emotion={"negative": True},
        )
        result = self.index.update(state, 0.0)

        self.assertEqual(
            [a["message"] for a in result["alerts"]],
            ["MICROSLEEP - WAKE UP", "SEATBELT NOT DETECTED",
             "NEGATIVE EMOTION"]
        )

    def test_unknown_seatbelt_is_not_penalised(self):
        result = self.index.update(
            make_state(seatbelt={"status": "UNKNOWN"}), 0.0
        )
        self.assertEqual(result["raw_score"], 100.0)

    def test_frequent_yawning(self):
        result = self.index.update(
            make_state(yawn={"yawn_count": 3, "recent_yawns": 3}), 0.0
        )
        self.assertIn("frequent_yawning", result["penalties"])


if __name__ == "__main__":
    unittest.main()

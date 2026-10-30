import os
import tempfile
import unittest

from evaluation import labels as labels_module
from evaluation.labels import Label, LabelError, parse, parse_file
from evaluation.matching import (
    match_points,
    match_intervals,
    spans_from_flags
)
from evaluation.evaluate import Recording, score_against, format_accuracy


class LabelParsingTest(unittest.TestCase):

    def test_reads_points_and_episodes(self):
        truth = parse([
            "start,end,label",
            "12.3,,blink",
            "20.0,26.5,distracted",
        ])

        self.assertEqual(len(truth), 2)
        self.assertTrue(truth[0].is_point)
        self.assertAlmostEqual(truth[0].start, 12.3)
        self.assertFalse(truth[1].is_point)
        self.assertAlmostEqual(truth[1].end, 26.5)

    def test_header_is_optional(self):
        self.assertEqual(len(parse(["1.0,,blink"])), 1)

    def test_comments_and_blank_lines_are_skipped(self):
        truth = parse([
            "# recorded 2026-10-30",
            "",
            "1.0,,blink",
            "   # a note",
        ])

        self.assertEqual(len(truth), 1)

    def test_rows_come_back_in_time_order(self):
        truth = parse(["9.0,,blink", "2.0,,blink", "5.0,,yawn"])

        self.assertEqual([label.start for label in truth], [2.0, 5.0, 9.0])

    def test_an_unknown_label_is_rejected(self):
        with self.assertRaises(LabelError) as caught:
            parse(["1.0,,sleepy"])

        self.assertIn("sleepy", str(caught.exception))

    def test_a_point_event_may_not_have_an_end(self):
        with self.assertRaises(LabelError):
            parse(["1.0,2.0,blink"])

    def test_an_episode_needs_an_end(self):
        with self.assertRaises(LabelError):
            parse(["1.0,,distracted"])

    def test_an_episode_may_not_end_before_it_starts(self):
        with self.assertRaises(LabelError):
            parse(["5.0,2.0,distracted"])

    def test_a_bad_number_names_its_line(self):
        with self.assertRaises(LabelError) as caught:
            parse(["1.0,,blink", "later,,blink"])

        self.assertIn("line 2", str(caught.exception))

    def test_a_short_row_is_rejected(self):
        with self.assertRaises(LabelError):
            parse(["1.0,blink"])

    def test_reads_from_a_file(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "truth.csv")

            with open(path, "w", encoding="utf-8") as handle:
                handle.write("start,end,label\n3.0,,yawn\n")

            truth = parse_file(path)

        self.assertEqual(truth[0].label, "yawn")

    def test_covers_only_applies_to_episodes(self):
        episode = Label("distracted", 2.0, 5.0)

        self.assertTrue(episode.covers(2.0))
        self.assertTrue(episode.covers(4.9))
        self.assertFalse(episode.covers(5.0))
        self.assertFalse(Label("blink", 2.0).covers(2.0))


class PointMatchingTest(unittest.TestCase):

    def labels(self, *times):
        return [Label("blink", t) for t in times]

    def test_a_perfect_match(self):
        score = match_points([1.0, 2.0], self.labels(1.0, 2.0), 0.5)

        self.assertEqual(score.true_positive, 2)
        self.assertEqual(score.precision, 1.0)
        self.assertEqual(score.recall, 1.0)
        self.assertEqual(score.f1, 1.0)

    def test_a_detection_slightly_late_still_counts(self):
        score = match_points([1.3], self.labels(1.0), 0.5)

        self.assertEqual(score.true_positive, 1)

    def test_a_detection_too_far_away_does_not(self):
        score = match_points([3.0], self.labels(1.0), 0.5)

        self.assertEqual(score.true_positive, 0)
        self.assertEqual(score.false_positive, 1)
        self.assertEqual(score.false_negative, 1)

    def test_each_label_is_used_once(self):
        """Two detections on one real blink is one hit and one invention."""

        score = match_points([1.0, 1.1], self.labels(1.0), 0.5)

        self.assertEqual(score.true_positive, 1)
        self.assertEqual(score.false_positive, 1)
        self.assertEqual(score.false_negative, 0)

    def test_nearest_first_keeps_close_labels_apart(self):
        """
        Pairing in time order would give the 1.0 label the 1.4
        detection and strand the 1.5 label, losing a real match.
        """

        score = match_points([1.4, 1.5], self.labels(1.0, 1.5), 0.5)

        self.assertEqual(score.true_positive, 2)

    def test_missing_everything(self):
        score = match_points([], self.labels(1.0, 2.0), 0.5)

        self.assertEqual(score.recall, 0.0)
        self.assertIsNone(score.precision)
        self.assertIsNone(score.f1)

    def test_detecting_nothing_real(self):
        score = match_points([1.0], [], 0.5)

        self.assertEqual(score.precision, 0.0)
        self.assertIsNone(score.recall)


class IntervalMatchingTest(unittest.TestCase):

    def test_agreement_over_time(self):
        samples = [float(i) for i in range(10)]
        truth = [Label("distracted", 2.0, 6.0)]

        score = match_intervals(samples, [(2.0, 6.0)], truth)

        self.assertEqual(score.true_positive, 4)
        self.assertEqual(score.false_positive, 0)
        self.assertEqual(score.false_negative, 0)

    def test_a_late_detection_costs_both_ways(self):
        samples = [float(i) for i in range(10)]
        truth = [Label("distracted", 2.0, 6.0)]

        score = match_intervals(samples, [(4.0, 8.0)], truth)

        self.assertEqual(score.true_positive, 2)   # 4, 5
        self.assertEqual(score.false_positive, 2)  # 6, 7
        self.assertEqual(score.false_negative, 2)  # 2, 3

    def test_unit_is_samples(self):
        score = match_intervals([0.0], [], [])

        self.assertEqual(score.unit, "samples")


class SpansTest(unittest.TestCase):

    def test_flags_become_spans(self):
        samples = [0.0, 1.0, 2.0, 3.0, 4.0]
        flags = [False, True, True, False, False]

        self.assertEqual(spans_from_flags(samples, flags), [(1.0, 3.0)])

    def test_a_span_running_to_the_end_is_kept(self):
        samples = [0.0, 1.0, 2.0]
        flags = [False, True, True]

        self.assertEqual(spans_from_flags(samples, flags), [(1.0, 2.0)])

    def test_nothing_flagged(self):
        self.assertEqual(spans_from_flags([0.0, 1.0], [False, False]), [])


class RecordingTest(unittest.TestCase):

    def state(self, blinks=0, distracted=False, phone=False):
        return {
            "blink": {"blink_count": blinks},
            "yawn": {"yawn_count": 0},
            "drowsiness": {"microsleep": False},
            "distraction": {"distracted": distracted},
            "phone": {"in_use": phone},
            "seatbelt": {"status": "ON"},
        }

    def record(self, frames):
        recording = Recording()

        for i, state in enumerate(frames):
            risk = {"score": 100.0}
            recording.add(state, float(i), risk, risk)

        return recording

    def test_a_rising_counter_marks_a_point_event(self):
        recording = self.record([
            self.state(blinks=0),
            self.state(blinks=1),
            self.state(blinks=1),
            self.state(blinks=2),
        ])

        self.assertEqual(recording.points["blink"], [1.0, 3.0])

    def test_a_counter_jumping_by_two_records_two(self):
        recording = self.record([self.state(0), self.state(2)])

        self.assertEqual(len(recording.points["blink"]), 2)

    def test_episodes_become_spans(self):
        recording = self.record([
            self.state(distracted=False),
            self.state(distracted=True),
            self.state(distracted=True),
            self.state(distracted=False),
        ])

        self.assertEqual(recording.spans("distracted"), [(1.0, 3.0)])

    def test_a_missing_monitor_does_not_crash(self):
        recording = Recording()
        recording.add({}, 0.0, {"score": 100.0}, {"score": 100.0})

        self.assertEqual(recording.points["blink"], [])
        self.assertEqual(recording.flags["distracted"], [False])


class ScoringAgainstTruthTest(unittest.TestCase):

    def build(self):
        recording = Recording()

        for i in range(10):
            state = {
                "blink": {"blink_count": 1 if i >= 2 else 0},
                "yawn": {"yawn_count": 0},
                "drowsiness": {"microsleep": False},
                "distraction": {"distracted": 3 <= i < 7},
                "phone": {"in_use": False},
                "seatbelt": {"status": "ON"},
            }
            risk = {"score": 100.0}
            recording.add(state, float(i), risk, risk)

        return recording

    def test_labels_are_scored_by_type(self):
        recording = self.build()
        truth = parse(["2.0,,blink", "3.0,7.0,distracted"])

        results = dict(score_against(recording, truth))

        self.assertEqual(results["blink"].true_positive, 1)
        self.assertEqual(results["distracted"].false_positive, 0)
        self.assertEqual(results["distracted"].recall, 1.0)

    def test_untouched_labels_are_left_out(self):
        results = dict(score_against(self.build(), parse(["2.0,,blink"])))

        self.assertIn("blink", results)
        self.assertNotIn("yawn", results)
        self.assertNotIn("no_seatbelt", results)

    def test_a_detection_with_no_label_is_still_reported(self):
        """Detecting something nobody labelled is a false positive, not
        a row to hide."""

        results = dict(score_against(self.build(), parse(["2.0,,blink"])))

        self.assertIn("distracted", results)
        self.assertEqual(results["distracted"].recall, None)
        self.assertEqual(results["distracted"].precision, 0.0)

    def test_the_report_renders(self):
        recording = self.build()
        truth = parse(["2.0,,blink", "3.0,7.0,distracted"])

        report = format_accuracy(score_against(recording, truth))

        self.assertIn("blink", report)
        self.assertIn("precision", report)

    def test_every_label_name_is_scoreable(self):
        for name in labels_module.LABELS:
            row = (
                f"1.0,,{name}"
                if name in labels_module.POINT_LABELS
                else f"1.0,2.0,{name}"
            )
            truth = parse([row])

            results = dict(score_against(self.build(), truth))

            self.assertIn(name, results)


if __name__ == "__main__":
    unittest.main()

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


class CoverDirectiveTest(unittest.TestCase):
    """
    Which labels a file claims to have gone looking for.

    Partial ground truth is the normal case, and without this the
    harness scores an unlabelled detector against zero and reports it
    at 0% precision - which reads as a broken detector rather than an
    unchecked one.
    """

    def test_no_directive_claims_everything(self):
        self.assertIsNone(parse(["1.0,,blink"]).covers)

    def test_a_directive_is_read(self):
        truth = parse([
            "#cover: distracted, phone",
            "20.0,26.5,distracted",
        ])

        self.assertEqual(truth.covers, frozenset({"distracted", "phone"}))

    def test_spaces_instead_of_commas_are_fine(self):
        truth = parse(["#cover: distracted phone", "20.0,26.5,distracted"])

        self.assertEqual(truth.covers, frozenset({"distracted", "phone"}))

    def test_case_does_not_matter(self):
        truth = parse(["#COVER: Distracted", "20.0,26.5,distracted"])

        self.assertEqual(truth.covers, frozenset({"distracted"}))

    def test_a_label_may_be_covered_without_ever_occurring(self):
        """
        The useful case: looked for, never happened. That is a real
        measurement of zero, not an absence of one.
        """

        truth = parse(["#cover: phone", "20.0,26.5,distracted"])

        self.assertEqual(truth.covers, frozenset({"phone"}))
        self.assertEqual(len(truth), 1)

    def test_an_unknown_name_is_rejected(self):
        with self.assertRaises(LabelError) as caught:
            parse(["#cover: sleepiness", "1.0,,blink"])

        self.assertIn("sleepiness", str(caught.exception))

    def test_an_empty_directive_is_rejected(self):
        """
        Ambiguous between "everything" and "nothing", so it is neither.
        """

        with self.assertRaises(LabelError):
            parse(["#cover:", "1.0,,blink"])

    def test_two_directives_are_rejected(self):
        with self.assertRaises(LabelError):
            parse(["#cover: phone", "#cover: blink", "1.0,,blink"])

    def test_it_is_still_a_list_of_labels(self):
        """Callers that only want the labels should not have to care."""

        truth = parse(["#cover: blink", "1.0,,blink", "2.0,,blink"])

        self.assertEqual(len(truth), 2)
        self.assertEqual(len(labels_module.of_type(truth, "blink")), 2)


class CoveredScoringTest(unittest.TestCase):

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

    def test_an_uncovered_label_is_not_scored(self):
        """
        Without the directive this same recording reports distracted at
        0% precision - see test_a_detection_with_no_label_is_still_
        reported, which is correct for complete ground truth.
        """

        truth = parse(["#cover: blink", "2.0,,blink"])
        results = dict(score_against(self.build(), truth))

        self.assertIn("blink", results)
        self.assertNotIn("distracted", results)

    def test_a_covered_label_is_scored_even_when_it_never_happened(self):
        truth = parse(["#cover: blink, distracted", "2.0,,blink"])
        results = dict(score_against(self.build(), truth))

        self.assertIn("distracted", results)
        self.assertEqual(results["distracted"].precision, 0.0)

    def test_the_report_names_what_went_unmeasured(self):
        """
        Silence would read as "the detector found nothing", which is a
        different claim from "nobody checked".
        """

        recording = self.build()
        truth = parse(["#cover: blink", "2.0,,blink"])

        report = format_accuracy(
            score_against(recording, truth), recording, truth.covers
        )

        self.assertIn("Not measured", report)
        self.assertIn("distracted", report)
        self.assertIn("1 detected", report)

    def test_nothing_is_named_when_nothing_was_detected(self):
        recording = self.build()
        truth = parse(["#cover: blink, distracted", "2.0,,blink"])

        report = format_accuracy(
            score_against(recording, truth), recording, truth.covers
        )

        self.assertNotIn("Not measured", report)

    def test_a_complete_file_names_nothing(self):
        recording = self.build()
        truth = parse(["2.0,,blink"])

        report = format_accuracy(
            score_against(recording, truth), recording, truth.covers
        )

        self.assertNotIn("Not measured", report)


class PerformedButUncoveredTest(unittest.TestCase):
    """
    A cue the file writes down and then excludes from scoring.

    take2.csv does exactly this with hands_off: it records the eleven
    second episode that was performed, and leaves the label out of the
    cover line because the clip was filmed at a desk and there is no
    wheel in it to have hands off. The monitor correctly reported
    nothing -- and a cue absent from the accuracy table *and* from the
    unmeasured list reads as a cue that never happened.
    """

    def quiet(self):
        """A recording in which the monitor reported nothing at all."""

        recording = Recording()

        for i in range(10):
            state = {
                "blink": {"blink_count": 0},
                "yawn": {"yawn_count": 0},
                "drowsiness": {"microsleep": False},
                "distraction": {"distracted": False},
                "phone": {"in_use": False},
                "wheel": {"hands_off": False},
                "seatbelt": {"status": "ON"},
            }
            risk = {"score": 100.0}
            recording.add(state, float(i), risk, risk)

        return recording

    def report(self, recording, truth):
        return format_accuracy(
            score_against(recording, truth), recording, truth.covers,
            truth
        )

    def test_an_excluded_episode_is_still_named(self):
        recording = self.quiet()
        truth = parse([
            "#cover: blink",
            "2.0,,blink",
            "3.0,6.0,hands_off",
        ])

        report = self.report(recording, truth)

        self.assertIn("Not measured", report)
        self.assertIn("hands_off (0 detected)", report)

    def test_an_excluded_point_event_is_too(self):
        recording = self.quiet()
        truth = parse([
            "#cover: blink",
            "2.0,,blink",
            "3.0,,yawn",
        ])

        report = self.report(recording, truth)

        self.assertIn("yawn (0 detected)", report)

    def test_a_label_nobody_wrote_down_is_not_named(self):
        """
        The rule this does not undo: silence about a cue that is in
        neither the file nor the detections is the right amount to say.
        """

        recording = self.quiet()
        truth = parse(["#cover: blink", "2.0,,blink"])

        report = self.report(recording, truth)

        self.assertNotIn("Not measured", report)

    def test_the_old_behaviour_survives_without_the_truth(self):
        """
        format_accuracy is called with the labels now, but callers that
        do not pass them still get everything that was detected.
        """

        recording = self.quiet()
        recording.flags["distracted"][0] = True
        truth = parse(["#cover: blink", "2.0,,blink"])

        report = format_accuracy(
            score_against(recording, truth), recording, truth.covers
        )

        self.assertIn("distracted", report)


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


def frame_state(blinks=0, distracted=False, phone=False):
    return {
        "blink": {"blink_count": blinks},
        "yawn": {"yawn_count": 0},
        "drowsiness": {"microsleep": False},
        "distraction": {"distracted": distracted},
        "phone": {"in_use": phone},
        "seatbelt": {"status": "ON"},
    }


def recorded(frames):
    recording = Recording()

    for i, state in enumerate(frames):
        risk = {"score": 100.0}
        recording.add(state, float(i), risk, risk)

    return recording


class RecordingTest(unittest.TestCase):

    state = staticmethod(frame_state)
    record = staticmethod(recorded)

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


class FaceGapTest(unittest.TestCase):
    """
    A detector reports None, not a dict, on a frame with no face. Its
    counter is untouched underneath, so the harness has to hold the
    total rather than read the absence as a count of zero.

    Found on a real recording: a clip with twenty blinks and several
    face gaps was reported as having a hundred and twelve, because
    every gap re-counted the whole history as one simultaneous burst.
    """

    state = staticmethod(frame_state)
    record = staticmethod(recorded)

    def blind(self):
        """What a frame with no face in it actually looks like."""

        state = frame_state()
        state["blink"] = None
        state["yawn"] = None

        return state

    def test_a_face_gap_invents_no_blinks(self):
        recording = self.record([
            self.state(blinks=20),
            self.blind(),
            self.blind(),
            self.state(blinks=20),
        ])

        self.assertEqual(recording.points["blink"], [0.0] * 20)

    def test_blinks_either_side_of_a_gap_count_once_each(self):
        recording = self.record([
            self.state(blinks=1),
            self.blind(),
            self.state(blinks=2),
        ])

        self.assertEqual(recording.points["blink"], [0.0, 2.0])

    def test_a_gap_at_the_very_start_is_not_a_burst(self):
        """
        The face is often missing for the first frames of a clip while
        the camera settles, which put the burst at t=0.
        """

        recording = self.record([
            self.blind(),
            self.blind(),
            self.state(blinks=3),
        ])

        self.assertEqual(recording.points["blink"], [2.0] * 3)

    def test_a_section_missing_entirely_is_also_held(self):
        """Older recordings predate some monitors."""

        partial = self.state(blinks=4)
        del partial["blink"]

        recording = self.record([self.state(blinks=4), partial,
                                 self.state(blinks=4)])

        self.assertEqual(recording.points["blink"], [0.0] * 4)


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

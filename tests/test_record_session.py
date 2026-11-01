"""
The guided recorder.

The camera cannot be driven in a test, so what is checked here is
everything either side of it: that the script covers what it claims to,
that frame boundaries become label times in the video's own timebase,
and that the file it writes is readable by the parser that has to read
it. The last of those is the one that would otherwise be found by hand,
two minutes after recording a session that cannot be scored.
"""

import unittest

from evaluation import labels as labels_module
from evaluation.labels import parse
from evaluation.record_session import (
    SCRIPT,
    SPEECH_LEAD,
    SPEECH_RATE,
    Step,
    Voice,
    build_labels,
    chime,
    covered_labels,
    format_labels,
    lead_time,
    total_seconds,
)


def boundaries_for(script, fps):
    """Frame indices as if every cue were held exactly as long as asked."""

    boundaries = []
    frame = 0

    for step in script:
        boundaries.append(frame)
        frame += int(round(step.seconds * fps))

    boundaries.append(frame)

    return boundaries


class ScriptTest(unittest.TestCase):

    def test_every_label_in_the_script_is_a_real_episode_label(self):
        for step in SCRIPT:
            if step.label is not None:
                self.assertIn(step.label, labels_module.INTERVAL_LABELS,
                              f"{step.label} is not an episode label")

    def test_it_covers_what_the_new_detectors_added(self):
        """
        The two features this harness has never been pointed at.
        """

        covered = covered_labels()

        self.assertIn("hands_off", covered)
        self.assertIn("camera_blocked", covered)

    def test_it_does_not_claim_to_cover_point_events(self):
        """
        A cue card cannot place a blink. Claiming otherwise would have
        the harness score them against labels nobody wrote.
        """

        for name in labels_module.POINT_LABELS:
            self.assertNotIn(name, covered_labels())

    def test_no_label_is_claimed_twice(self):
        covered = covered_labels()

        self.assertEqual(len(covered), len(set(covered)))

    def test_every_cue_is_long_enough_to_be_detected(self):
        """
        Each episode detector waits before it commits. A cue shorter
        than that dwell time would be scored as a miss for reasons that
        have nothing to do with accuracy.
        """

        for step in SCRIPT:
            if step.label is not None:
                self.assertGreaterEqual(step.seconds, 5.0, step.cue)

    def test_the_hands_cue_comes_after_the_camera_has_seen_hands(self):
        """
        The wheel monitor refuses to judge until it has seen hands.
        Cueing hands_off before that would label an episode the
        detector is designed never to report.
        """

        labels = [step.label for step in SCRIPT]
        cues = [step.cue.lower() for step in SCRIPT]

        hands_off = labels.index("hands_off")
        shown = next(i for i, cue in enumerate(cues) if "hold both hands" in cue)

        self.assertLess(shown, hands_off)

    def test_the_script_is_a_few_minutes_at_most(self):
        """Long enough to exercise everything, short enough to sit through."""

        self.assertGreater(total_seconds(), 60)
        self.assertLess(total_seconds(), 240)


class SpokenCueTest(unittest.TestCase):
    """
    The cues have to work with the screen unwatched.
    """

    def test_every_step_has_something_to_say(self):
        for step in SCRIPT:
            self.assertTrue(step.spoken.strip(), step.cue)

    def test_spoken_falls_back_to_the_written_cue(self):
        self.assertEqual(Step(2.0, "look ahead").spoken, "look ahead")

    def test_the_cues_that_matter_say_more_out_loud(self):
        """
        "Drop both hands" is ambiguous until you add "out of shot",
        and the banner has no room for it.
        """

        spoken = {step.label: step.spoken for step in SCRIPT if step.label}

        self.assertIn("out of the picture", spoken["hands_off"])
        self.assertIn("not the camera switch", spoken["camera_blocked"])

    def test_the_eyes_closed_cue_promises_to_say_when_to_stop(self):
        """
        Otherwise the only way to know is to open them and look.
        """

        spoken = next(s.spoken for s in SCRIPT if s.label == "microsleep")

        self.assertIn("until I tell you", spoken)

    def test_no_cue_tells_the_driver_to_act_now(self):
        """
        A cue is spoken before its step begins, so a cue containing
        "now" would be asking for the thing early. The word was in two
        of them when the cue was read out at the boundary instead.
        """

        for step in SCRIPT:
            self.assertNotIn(" now", step.spoken.lower(), step.cue)


class LeadTimeTest(unittest.TestCase):
    """
    When each cue has to start being spoken.

    The recording this script exists to produce had every episode
    labelled 2 to 4 seconds before it happened, because the cue was
    spoken at the moment the label began and the driver acted on the
    last word. The cues take 1.9 to 5.5 seconds to read aloud --
    measured by rendering them to audio -- which is the whole of that
    error.
    """

    def test_a_longer_cue_needs_longer(self):
        short = Step(10.0, "Look ahead.")
        long = Step(10.0, "Look ahead, and keep looking ahead, please.")

        self.assertGreater(lead_time(long), lead_time(short))

    def test_the_lead_covers_the_measured_speaking_rate(self):
        """
        The slowest of the script's real sentences ran at 13.5
        characters a second. Planning any faster than that would put
        the end of the cue inside the episode.
        """

        self.assertLessEqual(SPEECH_RATE, 13.5)

    def test_the_cue_is_meant_to_finish_before_the_step(self):
        self.assertGreater(SPEECH_LEAD, 0.0)

    def test_every_cue_fits_in_the_step_before_it(self):
        """
        A cue longer than the step it is announced during would be cut
        off at the start of that step. Nothing is broken if it is --
        the announcement just happens as early as it can -- but the
        driver then gets no quiet moment to prepare, so the script is
        written to avoid it.
        """

        for previous, step in zip(SCRIPT, SCRIPT[1:]):
            self.assertLessEqual(lead_time(step), previous.seconds,
                                 f"{step.cue!r} after {previous.cue!r}")


class VoiceTest(unittest.TestCase):
    """
    Speech is a convenience, never a dependency. Every failure here
    has to end in silence rather than in a lost recording.
    """

    def test_disabled_is_not_available(self):
        voice = Voice(enabled=False)

        self.assertFalse(voice.available)
        self.assertEqual(voice.reason, "turned off")

    def test_speaking_when_disabled_does_nothing(self):
        voice = Voice(enabled=False)

        voice.say("hello")
        voice.close()

    def test_a_dead_process_does_not_raise(self):
        class Dead:
            stdin = None

            def poll(self):
                return 1

        voice = Voice(enabled=False)
        voice.process = Dead()

        self.assertFalse(voice.available)
        voice.say("hello")

    def test_a_broken_pipe_stops_speech_rather_than_the_session(self):
        class Broken:
            class stdin:
                @staticmethod
                def write(text):
                    raise OSError("pipe closed")

                @staticmethod
                def flush():
                    pass

                @staticmethod
                def close():
                    pass

            def poll(self):
                return None

        voice = Voice(enabled=False)
        voice.process = Broken()

        voice.say("hello")

        self.assertFalse(voice.available)
        self.assertIn("stopped", voice.reason)

    def test_a_chime_never_raises(self):
        """It runs on a thread precisely so it cannot stall capture."""

        chime()


class LabelBuildingTest(unittest.TestCase):

    def test_times_come_from_frame_indices(self):
        """
        The harness reads a file's timestamps as frame / fps, so the
        labels have to be built the same way or they drift.
        """

        script = (
            Step(2.0, "ahead"),
            Step(4.0, "look away", "distracted"),
            Step(2.0, "ahead again"),
        )

        rows = build_labels(boundaries_for(script, 25.0), 25.0, script)

        self.assertEqual(len(rows), 1)

        start, end, label = rows[0]

        self.assertEqual(label, "distracted")
        self.assertAlmostEqual(start, 2.0)
        self.assertAlmostEqual(end, 6.0)

    def test_the_frame_rate_scales_the_times(self):
        script = (Step(4.0, "look away", "distracted"),)

        slow = build_labels(boundaries_for(script, 10.0), 10.0, script)
        fast = build_labels(boundaries_for(script, 30.0), 30.0, script)

        self.assertAlmostEqual(slow[0][1], fast[0][1], places=2)

    def test_unlabelled_steps_produce_no_rows(self):
        script = (Step(2.0, "ahead"), Step(2.0, "still ahead"))

        self.assertEqual(build_labels(boundaries_for(script, 20.0), 20.0,
                                      script), [])

    def test_a_recording_stopped_early_drops_the_cues_that_never_ran(self):
        """
        Q during the third cue must not leave labels claiming the
        fourth happened.
        """

        script = (
            Step(2.0, "ahead"),
            Step(4.0, "look away", "distracted"),
            Step(4.0, "phone up", "phone"),
        )

        # Stopped partway through the phone cue: no boundary for its end.
        rows = build_labels([0, 40, 120], 20.0, script)

        self.assertEqual([label for _, _, label in rows], ["distracted"])

    def test_a_zero_length_step_is_dropped(self):
        script = (Step(2.0, "ahead"), Step(4.0, "look away", "distracted"))

        rows = build_labels([0, 40, 40], 20.0, script)

        self.assertEqual(rows, [])


class LabelFileTest(unittest.TestCase):

    def written(self, complete=True):
        rows = build_labels(boundaries_for(SCRIPT, 20.0), 20.0)

        return format_labels(rows, covered_labels(), complete)

    def test_the_file_the_recorder_writes_parses(self):
        truth = parse(self.written().splitlines())

        self.assertEqual(len(truth), len(covered_labels()))
        self.assertEqual(truth.covers, frozenset(covered_labels()))

    def test_the_episodes_are_in_time_order_and_do_not_overlap(self):
        truth = parse(self.written().splitlines())

        for earlier, later in zip(truth, truth[1:]):
            self.assertLessEqual(earlier.end, later.start)

    def test_it_says_the_labels_are_cues_not_observations(self):
        """
        The caveat is the whole reason this file can be trusted at all:
        it records what the driver was asked to do.
        """

        text = self.written()

        self.assertIn("ASKED", text)
        self.assertIn("correct them", text)

    def test_it_says_when_the_recording_was_cut_short(self):
        self.assertIn("stopped early", self.written(complete=False))
        self.assertNotIn("stopped early", self.written(complete=True))

    def test_every_covered_label_appears_in_the_cover_line(self):
        text = self.written()
        line = next(l for l in text.splitlines()
                    if l.startswith("#cover:"))

        for name in covered_labels():
            self.assertIn(name, line)


if __name__ == "__main__":
    unittest.main()

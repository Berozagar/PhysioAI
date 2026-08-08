"""
Unit tests for the integration layer: run.PhysioSession, the overlay
Drawer, the synthetic pose generator, and the camera helpers.

No webcam, model file, or display is required.
"""

import numpy as np
import pytest

import run as app_run
from app.models.exercise_rules import EXERCISES, ExerciseRule
from app.utils import drawing
from app.utils.drawing import Drawer
from app.utils.synthetic import (
    NUM_POSE_LANDMARKS,
    create_pose_landmarks,
    cycle_for_rule,
)


@pytest.fixture
def session():
    """A session with audio disabled so no TTS engine is started."""
    return app_run.PhysioSession("shoulder_raise", audio=None, smoothing_alpha=None)


class TestSyntheticPoses:
    def test_produces_a_full_landmark_set(self):
        assert len(create_pose_landmarks(90.0)) == NUM_POSE_LANDMARKS

    def test_every_landmark_is_a_three_tuple(self):
        assert all(len(lm) == 3 for lm in create_pose_landmarks(90.0))

    @pytest.mark.parametrize("angle", [20.0, 90.0, 170.0])
    def test_requested_shoulder_angle_is_measurable(self, angle):
        from app.core.motion_analyzer import MotionAnalyzer

        angles = MotionAnalyzer().calculate_joint_angles(create_pose_landmarks(angle))
        assert angles["shoulder"] == pytest.approx(angle, abs=1.0)

    @pytest.mark.parametrize("knee", [80.0, 120.0, 180.0])
    def test_requested_knee_angle_is_measurable(self, knee):
        from app.core.motion_analyzer import MotionAnalyzer

        landmarks = create_pose_landmarks(180.0, knee_angle_deg=knee)
        angles = MotionAnalyzer().calculate_joint_angles(landmarks)
        assert angles["knee"] == pytest.approx(knee, abs=1.0)


class TestCycleGeneration:
    @pytest.mark.parametrize("name", sorted(EXERCISES))
    def test_sweep_crosses_both_thresholds_for_every_exercise(self, name):
        """
        Regression: the sweep used to stop exactly on the threshold, so
        smoothing lag meant knee_bend never registered a repetition.
        """
        rule = EXERCISES[name]
        angles = cycle_for_rule(rule["start_angle"], rule["end_angle"], repetitions=1)

        if rule["end_angle"] > rule["start_angle"]:
            assert max(angles) > rule["end_angle"]
            assert min(angles) < rule["start_angle"]
        else:
            assert min(angles) < rule["end_angle"]
            assert max(angles) > rule["start_angle"]

    @pytest.mark.parametrize("name", sorted(EXERCISES))
    def test_sweep_counts_the_requested_reps_through_the_real_rule(self, name):
        rule_config = EXERCISES[name]
        rule = ExerciseRule(name)
        for angle in cycle_for_rule(
            rule_config["start_angle"], rule_config["end_angle"], repetitions=3
        ):
            rule.update(angle)
        assert rule.get_count() == 3

    def test_angles_stay_anatomically_valid(self):
        for angle in cycle_for_rule(170.0, 80.0, repetitions=2):
            assert 0.0 <= angle <= 180.0

    def test_more_reps_produce_a_longer_sweep(self):
        assert len(cycle_for_rule(30.0, 160.0, 4)) > len(cycle_for_rule(30.0, 160.0, 2))


class TestSessionAnalysis:
    def test_analyze_adds_posture_keys(self, session):
        result = session.analyze(create_pose_landmarks(160.0))
        assert "posture_ok" in result and "posture_feedback" in result

    def test_good_form_is_reported_as_correct(self, session):
        result = session.analyze(create_pose_landmarks(160.0))
        assert result["posture_ok"] is True

    def test_poor_form_is_reported_with_messages(self, session):
        result = session.analyze(create_pose_landmarks(40.0))
        assert result["posture_ok"] is False
        assert result["posture_feedback"]

    def test_posture_is_skipped_when_required_joints_are_absent(self, session):
        ok, messages = session._check_posture({"shoulder": 160.0})  # no back/neck
        assert ok is True and messages == []

    def test_full_cycle_counts_a_repetition(self, session):
        for angle in cycle_for_rule(30.0, 160.0, repetitions=1):
            result = session.analyze(create_pose_landmarks(angle))
        assert result["repetitions"] == 1

    def test_smoothing_can_be_disabled(self):
        assert app_run.PhysioSession("squat", None, None).smoother is None

    def test_smoothing_is_enabled_when_alpha_given(self):
        assert app_run.PhysioSession("squat", None, 0.5).smoother is not None


class TestSessionLifecycle:
    def test_idle_result_reports_no_pose(self, session):
        assert session.idle_result()["feedback"] == app_run.NO_POSE_FEEDBACK

    def test_idle_result_preserves_the_rep_count(self, session):
        for angle in cycle_for_rule(30.0, 160.0, repetitions=1):
            session.analyze(create_pose_landmarks(angle))
        assert session.idle_result()["repetitions"] == 1

    def test_reset_clears_reps(self, session):
        for angle in cycle_for_rule(30.0, 160.0, repetitions=1):
            session.analyze(create_pose_landmarks(angle))
        session.reset()
        assert session.rule.get_count() == 0

    def test_toggle_mute_without_audio_is_a_no_op(self, session):
        assert session.toggle_mute() is False

    def test_unknown_exercise_registers_no_posture_joints(self):
        """Exercises absent from REQUIRED_POSTURE_JOINTS must degrade safely."""
        session = app_run.PhysioSession("shoulder_raise", None, None)
        session._required_joints = ()
        assert session._check_posture({"shoulder": 10.0}) == (True, [])


class TestKeyHandling:
    def test_q_requests_quit(self, session):
        assert app_run.handle_key(ord("q"), session) is False

    def test_escape_requests_quit(self, session):
        assert app_run.handle_key(27, session) is False

    def test_r_resets_without_quitting(self, session):
        for angle in cycle_for_rule(30.0, 160.0, repetitions=1):
            session.analyze(create_pose_landmarks(angle))
        assert app_run.handle_key(ord("r"), session) is True
        assert session.rule.get_count() == 0

    def test_unknown_key_is_ignored(self, session):
        assert app_run.handle_key(ord("z"), session) is True

    def test_no_key_pressed_is_ignored(self, session):
        assert app_run.handle_key(255, session) is True


class TestOverlay:
    @pytest.fixture
    def frame(self):
        return np.zeros((720, 1280, 3), dtype=np.uint8)

    def test_returns_a_frame_of_the_same_shape(self, frame):
        result = Drawer().draw_information(
            frame, "shoulder_raise", 3, "up", 55.0, "Perfect!", 30.0
        )
        assert result.shape == (720, 1280, 3)

    def test_draws_something_onto_the_frame(self, frame):
        Drawer().draw_information(frame, "squat", 1, "down", 10.0, "Go lower", 30.0)
        assert frame.any()

    def test_handles_missing_progress(self, frame):
        """Regression: MotionAnalyzer returns progress=None without a rule."""
        Drawer().draw_information(frame, "squat", 0, None, None, "Waiting", 30.0)
        assert frame.any()

    def test_posture_lines_stay_inside_the_panel(self, frame):
        """
        Regression: the posture block used to be drawn below the panel
        background because the heading row was not counted.
        """
        lines = ["Raise your arm higher.", "Back posture correct.", "Head ok."]
        Drawer().draw_information(
            frame, "shoulder_raise", 2, "down", 0.0, "Raise your arm higher",
            30.0, posture_feedback=lines, posture_ok=False,
        )

        panel_height = (
            drawing.PANEL_BASE_HEIGHT
            + drawing.PANEL_LINE_HEIGHT * (len(lines) + 1)
            + drawing.PANEL_BOTTOM_PADDING
        )
        last_line_y = 272 + drawing.PANEL_LINE_HEIGHT * len(lines)

        assert last_line_y < panel_height, "last posture line falls outside the panel"
        # x=5 is inside the panel but clear of any glyphs.
        assert tuple(frame[last_line_y, 5]) == drawing.COLOR_PANEL

    def test_panel_grows_with_more_posture_lines(self, frame):
        drawer = Drawer()
        short = np.zeros_like(frame)
        drawer.draw_information(short, "knee_bend", 0, None, 0.0, "x", 30.0,
                                posture_feedback=["one"], posture_ok=True)
        tall = np.zeros_like(frame)
        drawer.draw_information(tall, "shoulder_raise", 0, None, 0.0, "x", 30.0,
                                posture_feedback=["one", "two", "three"], posture_ok=True)
        assert int(tall.any(axis=(1, 2)).sum()) > int(short.any(axis=(1, 2)).sum())

    def test_long_feedback_is_truncated(self):
        assert Drawer()._truncate("x" * 200).endswith("...")

    def test_short_feedback_is_left_alone(self):
        assert Drawer()._truncate("Perfect!") == "Perfect!"

    def test_non_string_feedback_does_not_crash(self):
        assert Drawer()._truncate(None) == "None"


class TestCameraHelpers:
    def test_at_least_one_backend_is_offered(self):
        assert len(app_run.candidate_backends()) >= 1

    def test_backends_are_id_and_name_pairs(self):
        for backend, name in app_run.candidate_backends():
            assert isinstance(backend, int) and isinstance(name, str)

    def test_missing_video_file_returns_none(self):
        assert app_run.open_capture("no_such_file.mp4", 640, 480) is None


class TestArgumentParsing:
    def test_defaults_are_sane(self):
        args = app_run.parse_args([])
        assert args.exercise == app_run.DEFAULT_EXERCISE
        assert args.source == "0"
        assert args.self_test is False

    def test_exercise_choice_is_validated(self):
        with pytest.raises(SystemExit):
            app_run.parse_args(["--exercise", "moon_walk"])

    def test_flags_are_parsed(self):
        args = app_run.parse_args(["--headless", "--no-audio", "--max-frames", "5"])
        assert args.headless and args.no_audio and args.max_frames == 5

    def test_zero_smoothing_disables_smoothing(self):
        """main() converts 0 to None because the smoother requires (0, 1]."""
        assert app_run.parse_args(["--smoothing", "0"]).smoothing == 0.0

    def test_list_exercises_exits_cleanly(self, capsys):
        assert app_run.main(["--list-exercises"]) == 0
        assert "shoulder_raise" in capsys.readouterr().out

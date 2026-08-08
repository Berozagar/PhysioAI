"""Unit tests for app/core/motion_analyzer.py."""

import pytest

from app.core.motion_analyzer import AnalyzerConfig, MotionAnalyzer
from app.models.exercise_rules import ExerciseRule
from app.utils.synthetic import create_pose_landmarks

RESULT_KEYS = {
    "angles", "velocity", "acceleration", "smoothness", "repetitions",
    "movement_quality", "range_of_motion", "in_motion", "paused",
    "feedback", "stage", "progress", "exercise",
}


class Landmark:
    """Stand-in for a MediaPipe NormalizedLandmark."""

    def __init__(self, x, y, z=0.0):
        self.x = x
        self.y = y
        self.z = z


class TestNormalizeLandmarks:
    def test_passes_three_dimensional_tuples_through(self):
        assert MotionAnalyzer._normalize_landmarks([(1, 2, 3)]) == [(1.0, 2.0, 3.0)]

    def test_pads_two_dimensional_tuples_with_zero_z(self):
        assert MotionAnalyzer._normalize_landmarks([(1, 2)]) == [(1.0, 2.0, 0.0)]

    def test_reads_attribute_style_landmarks(self):
        assert MotionAnalyzer._normalize_landmarks([Landmark(1, 2, 3)]) == [(1.0, 2.0, 3.0)]

    def test_defaults_z_when_object_has_no_z(self):
        class Flat:
            x, y = 1.0, 2.0

        assert MotionAnalyzer._normalize_landmarks([Flat()]) == [(1.0, 2.0, 0.0)]

    def test_rejects_unsupported_type(self):
        with pytest.raises(TypeError, match="Unsupported"):
            MotionAnalyzer._normalize_landmarks([object()])

    def test_rejects_single_element_tuple(self):
        with pytest.raises(ValueError, match="invalid length"):
            MotionAnalyzer._normalize_landmarks([(1,)])


class TestJointAngles:
    def test_computes_every_configured_joint(self):
        analyzer = MotionAnalyzer()
        angles = analyzer.calculate_joint_angles(create_pose_landmarks(90.0))
        assert set(angles) == set(AnalyzerConfig().joint_config)

    def test_measures_the_requested_shoulder_angle(self):
        analyzer = MotionAnalyzer()
        angles = analyzer.calculate_joint_angles(create_pose_landmarks(90.0))
        assert angles["shoulder"] == pytest.approx(90.0, abs=1.0)

    def test_straight_leg_reads_as_180(self):
        analyzer = MotionAnalyzer()
        angles = analyzer.calculate_joint_angles(create_pose_landmarks(45.0))
        assert angles["knee"] == pytest.approx(180.0, abs=1.0)

    def test_skips_joints_when_landmarks_are_missing(self):
        analyzer = MotionAnalyzer()
        # Only 5 landmarks: every configured triplet needs higher indices.
        assert analyzer.calculate_joint_angles([(0.5, 0.5, 0.0)] * 5) == {}


class TestVelocityAndPauses:
    def test_first_frame_has_zero_velocity(self):
        analyzer = MotionAnalyzer()
        assert analyzer.analyze_frame(create_pose_landmarks(45.0), timestamp=1.0)[
            "velocity"
        ] == 0.0

    def test_movement_between_frames_produces_velocity(self):
        analyzer = MotionAnalyzer()
        analyzer.analyze_frame(create_pose_landmarks(45.0, nose_y=0.20), timestamp=1.0)
        result = analyzer.analyze_frame(
            create_pose_landmarks(45.0, nose_y=0.40), timestamp=1.5
        )
        assert result["velocity"] > 0.0

    def test_duplicate_timestamp_does_not_divide_by_zero(self):
        analyzer = MotionAnalyzer()
        analyzer.analyze_frame(create_pose_landmarks(45.0), timestamp=1.0)
        result = analyzer.analyze_frame(create_pose_landmarks(90.0), timestamp=1.0)
        assert result["velocity"] == 0.0

    def test_detect_motion_uses_configured_threshold(self):
        analyzer = MotionAnalyzer(AnalyzerConfig(motion_threshold=0.5))
        assert analyzer.detect_motion(0.6) is True
        assert analyzer.detect_motion(0.4) is False

    def test_sustained_stillness_is_reported_as_paused(self):
        analyzer = MotionAnalyzer(
            AnalyzerConfig(pause_velocity_threshold=1.0, pause_duration_threshold=0.5)
        )
        assert analyzer.detect_pauses(0.0, timestamp=10.0) is False  # pause starts
        assert analyzer.detect_pauses(0.0, timestamp=10.6) is True   # long enough

    def test_movement_cancels_a_pause(self):
        analyzer = MotionAnalyzer(
            AnalyzerConfig(pause_velocity_threshold=1.0, pause_duration_threshold=0.5)
        )
        analyzer.detect_pauses(0.0, timestamp=10.0)
        assert analyzer.detect_pauses(5.0, timestamp=10.6) is False


class TestAnalyzeFrame:
    def test_returns_the_documented_keys(self):
        result = MotionAnalyzer().analyze_frame(create_pose_landmarks(45.0))
        assert RESULT_KEYS.issubset(result)

    def test_bad_input_returns_error_dict_instead_of_raising(self):
        """A single bad frame must never crash the video loop."""
        result = MotionAnalyzer().analyze_frame(["not a landmark"])
        assert "error" in result
        assert result["feedback"] == "Analysis error."

    def test_exercise_rule_supplies_stage_and_exercise_name(self):
        analyzer = MotionAnalyzer()
        rule = ExerciseRule("shoulder_raise")
        result = analyzer.analyze_frame(create_pose_landmarks(165.0), exercise_rule=rule)
        assert result["exercise"] == "shoulder_raise"
        assert result["stage"] == "up"

    def test_rule_drives_repetitions_across_a_full_cycle(self):
        analyzer = MotionAnalyzer()
        rule = ExerciseRule("shoulder_raise")
        for angle in (20.0, 90.0, 165.0, 90.0, 20.0):
            result = analyzer.analyze_frame(
                create_pose_landmarks(angle), exercise_rule=rule
            )
        assert result["repetitions"] == 1

    def test_without_a_rule_stage_and_exercise_stay_none(self):
        result = MotionAnalyzer().analyze_frame(create_pose_landmarks(45.0))
        assert result["stage"] is None
        assert result["exercise"] is None

    def test_generic_feedback_is_produced_without_a_rule(self):
        result = MotionAnalyzer().analyze_frame(create_pose_landmarks(45.0))
        assert isinstance(result["feedback"], str) and result["feedback"]


class TestGenericRepetitionCounter:
    def test_confirmed_direction_reversals_count_reps(self):
        analyzer = MotionAnalyzer(
            AnalyzerConfig(rep_angle_threshold=10.0, min_direction_confirm_frames=1)
        )
        for angle in (0.0, 50.0, 100.0, 50.0, 0.0):
            count = analyzer.count_repetitions(angle)
        assert count >= 1

    def test_small_jitter_never_counts(self):
        analyzer = MotionAnalyzer(AnalyzerConfig(rep_angle_threshold=20.0))
        for angle in (90.0, 92.0, 89.0, 91.0, 90.5):
            count = analyzer.count_repetitions(angle)
        assert count == 0


class TestSessionState:
    def test_range_of_motion_is_none_for_unseen_joint(self):
        assert MotionAnalyzer().evaluate_range_of_motion("shoulder") is None

    def test_range_of_motion_grows_over_a_sweep(self):
        analyzer = MotionAnalyzer()
        for angle in (20.0, 90.0, 165.0):
            analyzer.analyze_frame(create_pose_landmarks(angle))
        rom = analyzer.evaluate_range_of_motion("shoulder")
        assert rom is not None and rom.range_of_motion > 100.0

    def test_reset_clears_history_and_counters(self):
        analyzer = MotionAnalyzer()
        for angle in (20.0, 90.0, 165.0):
            analyzer.analyze_frame(create_pose_landmarks(angle))

        analyzer.reset()

        assert analyzer.evaluate_range_of_motion("shoulder") is None
        assert analyzer.analyze_frame(create_pose_landmarks(45.0))["velocity"] == 0.0

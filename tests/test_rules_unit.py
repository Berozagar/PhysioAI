"""Unit tests for app/models/exercise_rules.py and posture_rules.py."""

import pytest

from app.models.exercise_rules import EXERCISES, ExerciseRule
from app.models.posture_rules import PostureRules


def drive(rule: ExerciseRule, angles) -> int:
    """Feed a sequence of angles through the rule and return the rep count."""
    for angle in angles:
        rule.update(angle)
    return rule.get_count()


class TestExerciseRuleConstruction:
    def test_rejects_unknown_exercise(self):
        with pytest.raises(ValueError, match="not supported"):
            ExerciseRule("moon_walk")

    @pytest.mark.parametrize("name", sorted(EXERCISES))
    def test_every_configured_exercise_can_be_built(self, name):
        rule = ExerciseRule(name)
        assert rule.get_count() == 0
        assert rule.get_stage() is None

    @pytest.mark.parametrize("name", sorted(EXERCISES))
    def test_every_rule_declares_required_config_keys(self, name):
        rule = EXERCISES[name]
        for key in ("joint", "start_angle", "end_angle", "start_stage", "end_stage"):
            assert key in rule, f"{name} is missing {key}"
        for tone in ("low", "good", "high"):
            assert tone in rule["feedback"], f"{name} feedback is missing {tone}"


class TestRepetitionCounting:
    def test_increasing_exercise_counts_one_full_cycle(self):
        rule = ExerciseRule("shoulder_raise")
        assert drive(rule, [20, 90, 165, 90, 20]) == 1

    def test_decreasing_exercise_counts_one_full_cycle(self):
        rule = ExerciseRule("squat")
        assert drive(rule, [175, 120, 85, 120, 175]) == 1

    def test_partial_movement_does_not_count(self):
        rule = ExerciseRule("shoulder_raise")
        # Never reaches end_angle (160), so no rep completes.
        assert drive(rule, [20, 90, 120, 90, 20]) == 0

    def test_reaching_top_without_returning_does_not_count(self):
        rule = ExerciseRule("shoulder_raise")
        assert drive(rule, [20, 90, 165]) == 0
        assert rule.get_stage() == "up"

    def test_multiple_cycles_accumulate(self):
        rule = ExerciseRule("shoulder_raise")
        assert drive(rule, [20, 165, 20, 165, 20, 165, 20]) == 3

    def test_reset_clears_count_and_stage(self):
        rule = ExerciseRule("shoulder_raise")
        drive(rule, [20, 165, 20])
        rule.reset()
        assert rule.get_count() == 0
        assert rule.get_stage() is None


class TestProgress:
    def test_at_start_angle_progress_is_zero(self):
        assert ExerciseRule("shoulder_raise").get_progress(30) == pytest.approx(0.0)

    def test_at_end_angle_progress_is_full(self):
        assert ExerciseRule("shoulder_raise").get_progress(160) == pytest.approx(100.0)

    @pytest.mark.parametrize("angle", [-50, 0, 20, 200, 400])
    def test_progress_is_clamped_to_0_100(self, angle):
        assert 0.0 <= ExerciseRule("shoulder_raise").get_progress(angle) <= 100.0

    def test_decreasing_exercise_progress_increases_as_angle_drops(self):
        rule = ExerciseRule("squat")
        assert rule.get_progress(90) > rule.get_progress(170)

    def test_zero_span_rule_does_not_divide_by_zero(self):
        """Regression: a rule with start_angle == end_angle used to raise."""
        rule = ExerciseRule("shoulder_raise")
        rule.rule = dict(rule.rule, start_angle=90, end_angle=90)
        assert rule.get_progress(90) == 0.0


class TestFeedback:
    def test_increasing_exercise_below_target_says_go_higher(self):
        assert ExerciseRule("shoulder_raise").get_feedback(60) == "Raise your arm higher"

    def test_increasing_exercise_at_target_is_positive(self):
        assert ExerciseRule("shoulder_raise").get_feedback(160) == "Perfect!"

    def test_increasing_exercise_past_target_warns(self):
        assert ExerciseRule("shoulder_raise").get_feedback(179) == (
            "Do not overextend your shoulder"
        )

    def test_decreasing_exercise_above_target_says_go_lower(self):
        assert ExerciseRule("squat").get_feedback(170) == "Go lower"

    def test_decreasing_exercise_at_target_is_positive(self):
        assert ExerciseRule("squat").get_feedback(90) == "Nice squat!"


class TestGetStatus:
    def test_returns_all_expected_keys(self):
        status = ExerciseRule("shoulder_raise").get_status(90)
        assert set(status) == {"exercise", "counter", "stage", "progress", "feedback"}

    def test_reports_the_requested_exercise(self):
        assert ExerciseRule("squat").get_status(120)["exercise"] == "squat"

    def test_advances_state_like_update(self):
        rule = ExerciseRule("shoulder_raise")
        for angle in (20, 165, 20):
            rule.get_status(angle)
        assert rule.get_count() == 1


class TestPostureChecks:
    @pytest.fixture
    def rules(self):
        return PostureRules()

    @pytest.mark.parametrize(
        "angle, expected", [(120, False), (160, True), (179, False)]
    )
    def test_shoulder_raise_band(self, rules, angle, expected):
        assert rules.check_shoulder_raise(angle)[0] is expected

    @pytest.mark.parametrize("angle, expected", [(150, False), (100, True), (50, False)])
    def test_squat_band(self, rules, angle, expected):
        assert rules.check_squat(angle)[0] is expected

    @pytest.mark.parametrize("angle, expected", [(150, False), (80, True), (40, False)])
    def test_knee_bend_band(self, rules, angle, expected):
        assert rules.check_knee_bend(angle)[0] is expected

    @pytest.mark.parametrize("angle, expected", [(120, False), (175, True)])
    def test_back_alignment(self, rules, angle, expected):
        assert rules.check_back_alignment(angle)[0] is expected

    @pytest.mark.parametrize("angle, expected", [(120, False), (170, True)])
    def test_head_alignment(self, rules, angle, expected):
        assert rules.check_head_alignment(angle)[0] is expected

    def test_every_check_returns_a_message(self, rules):
        for _, message in (
            rules.check_squat(100),
            rules.check_knee_bend(80),
            rules.check_back_alignment(90),
        ):
            assert isinstance(message, str) and message


class TestPostureEvaluate:
    @pytest.fixture
    def rules(self):
        return PostureRules()

    def test_good_shoulder_raise_form_passes(self, rules):
        verdict = rules.evaluate(
            "shoulder_raise", {"shoulder": 160, "back": 175, "neck": 170}
        )
        assert verdict["correct"] is True
        assert len(verdict["feedback"]) == 3

    def test_bad_back_fails_shoulder_raise(self, rules):
        verdict = rules.evaluate(
            "shoulder_raise", {"shoulder": 160, "back": 120, "neck": 170}
        )
        assert verdict["correct"] is False
        assert "Straighten your back." in verdict["feedback"]

    def test_good_squat_form_passes(self, rules):
        verdict = rules.evaluate("squat", {"knee": 100, "back": 175})
        assert verdict["correct"] is True

    def test_good_knee_bend_form_passes(self, rules):
        verdict = rules.evaluate("knee_bend", {"knee": 80})
        assert verdict["correct"] is True

    def test_unsupported_exercise_is_reported_not_raised(self, rules):
        verdict = rules.evaluate("moon_walk", {})
        assert verdict["correct"] is False
        assert verdict["feedback"] == ["Unsupported exercise."]

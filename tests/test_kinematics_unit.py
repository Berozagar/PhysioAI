"""Unit tests for app/core/kinematics.py -- the pure math layer."""

import pytest

from app.core.kinematics import (
    ROMResult,
    back_angle,
    calculate_acceleration,
    calculate_angle,
    calculate_range_of_motion,
    calculate_smoothness,
    calculate_velocity,
    elbow_angle,
    euclidean_distance,
    knee_angle,
    moving_average,
    neck_angle,
    shoulder_angle,
    smooth_series,
)


class TestEuclideanDistance:
    def test_returns_hypotenuse_for_3_4_5_triangle(self):
        assert euclidean_distance((0, 0, 0), (3, 4, 0)) == pytest.approx(5.0)

    def test_works_in_two_dimensions(self):
        assert euclidean_distance((0, 0), (0, 2)) == pytest.approx(2.0)

    def test_distance_to_self_is_zero(self):
        assert euclidean_distance((1.5, 2.5), (1.5, 2.5)) == pytest.approx(0.0)

    def test_raises_when_dimensions_differ(self):
        with pytest.raises(ValueError, match="dimensionality"):
            euclidean_distance((0, 0), (1, 1, 1))


class TestCalculateAngle:
    def test_perpendicular_rays_give_90_degrees(self):
        assert calculate_angle((0, 1), (0, 0), (1, 0)) == pytest.approx(90.0)

    def test_opposite_rays_give_180_degrees(self):
        assert calculate_angle((0, 1), (0, 0), (0, -1)) == pytest.approx(180.0)

    def test_identical_rays_give_0_degrees(self):
        assert calculate_angle((0, 1), (0, 0), (0, 2)) == pytest.approx(0.0)

    def test_result_never_exceeds_180(self):
        # Floating point can push cosine outside [-1, 1]; the implementation
        # clips before arccos so this must not become NaN.
        angle = calculate_angle((1e-9, 1.0), (0, 0), (-1e-9, 1.0))
        assert 0.0 <= angle <= 180.0

    def test_raises_when_vertex_coincides_with_endpoint(self):
        with pytest.raises(ValueError, match="Degenerate"):
            calculate_angle((0, 0), (0, 0), (1, 0))


class TestJointHelpers:
    """Each helper is calculate_angle with a documented vertex."""

    def test_straight_arm_elbow_is_180(self):
        assert elbow_angle((0, 0), (0, 1), (0, 2)) == pytest.approx(180.0)

    def test_arm_at_side_shoulder_is_0(self):
        # Hip below shoulder, elbow also below shoulder -> rays coincide.
        assert shoulder_angle((0, 2), (0, 0), (0, 1)) == pytest.approx(0.0)

    def test_straight_leg_knee_is_180(self):
        assert knee_angle((0, 0), (0, 1), (0, 2)) == pytest.approx(180.0)

    def test_neutral_spine_back_is_180(self):
        assert back_angle((0, 0), (0, 1), (0, 2)) == pytest.approx(180.0)

    def test_stacked_head_neck_is_180(self):
        assert neck_angle((0, 0), (0, 1), (0, 2)) == pytest.approx(180.0)


class TestVelocityAndAcceleration:
    def test_velocity_is_distance_over_time(self):
        assert calculate_velocity((0, 0), (0, 1), 0.5) == pytest.approx(2.0)

    def test_acceleration_is_velocity_change_over_time(self):
        assert calculate_acceleration(1.0, 3.0, 0.5) == pytest.approx(4.0)

    def test_deceleration_is_negative(self):
        assert calculate_acceleration(3.0, 1.0, 0.5) == pytest.approx(-4.0)

    @pytest.mark.parametrize("bad_dt", [0.0, -1.0])
    def test_velocity_rejects_non_positive_delta_time(self, bad_dt):
        with pytest.raises(ValueError, match="positive"):
            calculate_velocity((0, 0), (0, 1), bad_dt)

    @pytest.mark.parametrize("bad_dt", [0.0, -1.0])
    def test_acceleration_rejects_non_positive_delta_time(self, bad_dt):
        with pytest.raises(ValueError, match="positive"):
            calculate_acceleration(1.0, 2.0, bad_dt)


class TestSmoothing:
    def test_moving_average_uses_only_trailing_window(self):
        assert moving_average([1, 2, 3, 4, 5], window_size=3) == pytest.approx(4.0)

    def test_moving_average_of_empty_series_is_zero(self):
        assert moving_average([]) == 0.0

    def test_moving_average_rejects_non_positive_window(self):
        with pytest.raises(ValueError, match="positive"):
            moving_average([1, 2], window_size=0)

    def test_smooth_series_matches_sliding_window_means(self):
        assert smooth_series([1, 2, 3, 4, 5], window_size=2) == pytest.approx(
            [1.0, 1.5, 2.5, 3.5, 4.5]
        )

    def test_smooth_series_preserves_length(self):
        values = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0]
        assert len(smooth_series(values, window_size=3)) == len(values)

    def test_smooth_series_of_empty_is_empty(self):
        assert smooth_series([]) == []


class TestRangeOfMotion:
    def test_reports_min_max_and_span(self):
        result = calculate_range_of_motion([30, 45, 90, 60])
        assert isinstance(result, ROMResult)
        assert result.min_angle == pytest.approx(30.0)
        assert result.max_angle == pytest.approx(90.0)
        assert result.range_of_motion == pytest.approx(60.0)

    def test_constant_series_has_zero_range(self):
        assert calculate_range_of_motion([90, 90]).range_of_motion == pytest.approx(0.0)

    def test_raises_on_empty_sequence(self):
        with pytest.raises(ValueError, match="at least one"):
            calculate_range_of_motion([])


class TestMovementSmoothness:
    def test_steady_velocity_scores_high(self):
        assert calculate_smoothness([0.10, 0.11, 0.10, 0.09, 0.10]) > 90.0

    def test_erratic_velocity_scores_low(self):
        assert calculate_smoothness([0.01, 5.0, 0.02, 4.0, 0.01]) < 30.0

    def test_needs_at_least_two_samples(self):
        assert calculate_smoothness([0.5]) == 0.0
        assert calculate_smoothness([]) == 0.0

    def test_score_stays_within_bounds(self):
        assert 0.0 <= calculate_smoothness([0.0, 100.0, 0.0]) <= 100.0

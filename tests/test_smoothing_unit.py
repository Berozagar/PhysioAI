"""
Unit tests for app/utils/smoothing.py.

Several of these lock in behaviour that was previously broken: the
smoother used to mutate the caller's landmark objects in place, and it
only accepted objects exposing .x/.y/.z -- so it could not consume the
plain tuples PoseDetector.get_landmarks() actually returns.
"""

import pytest

from app.utils.smoothing import DEFAULT_ALPHA, LandmarkSmoother, SmoothedLandmark


class MutablePoint:
    """Stand-in for a MediaPipe NormalizedLandmark."""

    def __init__(self, x, y, z=0.0):
        self.x = x
        self.y = y
        self.z = z


class TestConstruction:
    def test_defaults_to_documented_alpha(self):
        assert LandmarkSmoother().alpha == DEFAULT_ALPHA

    @pytest.mark.parametrize("bad_alpha", [0.0, -0.5, 1.5])
    def test_rejects_alpha_outside_unit_interval(self, bad_alpha):
        with pytest.raises(ValueError, match="alpha"):
            LandmarkSmoother(alpha=bad_alpha)

    def test_alpha_of_one_is_allowed(self):
        assert LandmarkSmoother(alpha=1.0).alpha == 1.0


class TestInputHandling:
    def test_accepts_plain_xyz_tuples(self):
        """Regression: tuples from PoseDetector.get_landmarks() must work."""
        result = LandmarkSmoother().smooth([(0.1, 0.2, 0.3)])
        assert result == [SmoothedLandmark(0.1, 0.2, 0.3)]

    def test_accepts_two_dimensional_tuples_and_defaults_z(self):
        assert LandmarkSmoother().smooth([(0.1, 0.2)])[0].z == 0.0

    def test_accepts_mediapipe_style_objects(self):
        result = LandmarkSmoother().smooth([MutablePoint(1.0, 2.0, 3.0)])
        assert (result[0].x, result[0].y, result[0].z) == (1.0, 2.0, 3.0)

    def test_none_passes_through(self):
        assert LandmarkSmoother().smooth(None) is None

    def test_empty_frame_returns_empty_list(self):
        assert LandmarkSmoother().smooth([]) == []

    def test_rejects_unsupported_landmark_type(self):
        with pytest.raises(TypeError, match="Unsupported"):
            LandmarkSmoother().smooth(["not a landmark"])

    def test_rejects_too_short_tuple(self):
        with pytest.raises(ValueError, match="invalid length"):
            LandmarkSmoother().smooth([(0.5,)])


class TestImmutability:
    def test_does_not_mutate_input_objects(self):
        """Regression: smoothing used to overwrite curr.x/.y/.z in place."""
        smoother = LandmarkSmoother(alpha=0.5)
        smoother.smooth([MutablePoint(0.0, 0.0, 0.0)])

        second = MutablePoint(10.0, 20.0, 30.0)
        smoother.smooth([second])

        assert (second.x, second.y, second.z) == (10.0, 20.0, 30.0)

    def test_does_not_mutate_input_tuples(self):
        smoother = LandmarkSmoother(alpha=0.5)
        smoother.smooth([(0.0, 0.0, 0.0)])

        frame = [(10.0, 20.0, 30.0)]
        smoother.smooth(frame)

        assert frame == [(10.0, 20.0, 30.0)]

    def test_result_is_an_immutable_namedtuple(self):
        landmark = LandmarkSmoother().smooth([(1.0, 2.0, 3.0)])[0]
        with pytest.raises(AttributeError):
            landmark.x = 99.0

    def test_result_supports_both_tuple_and_attribute_access(self):
        landmark = LandmarkSmoother().smooth([(1.0, 2.0, 3.0)])[0]
        assert landmark[0] == landmark.x
        x, y, z = landmark
        assert (x, y, z) == (1.0, 2.0, 3.0)


class TestExponentialAverage:
    def test_first_frame_is_returned_unsmoothed(self):
        assert LandmarkSmoother(alpha=0.5).smooth([(10.0, 20.0, 0.0)])[0].x == 10.0

    def test_second_frame_blends_by_alpha(self):
        smoother = LandmarkSmoother(alpha=0.5)
        smoother.smooth([(10.0, 20.0, 0.0)])
        blended = smoother.smooth([(20.0, 30.0, 0.0)])[0]
        assert (blended.x, blended.y) == pytest.approx((15.0, 25.0))

    def test_alpha_of_one_tracks_input_exactly(self):
        smoother = LandmarkSmoother(alpha=1.0)
        smoother.smooth([(0.0, 0.0, 0.0)])
        assert smoother.smooth([(9.0, 9.0, 9.0)])[0].x == pytest.approx(9.0)

    def test_lower_alpha_lags_further_behind(self):
        slow, fast = LandmarkSmoother(alpha=0.1), LandmarkSmoother(alpha=0.9)
        for smoother in (slow, fast):
            smoother.smooth([(0.0, 0.0, 0.0)])

        slow_x = slow.smooth([(100.0, 0.0, 0.0)])[0].x
        fast_x = fast.smooth([(100.0, 0.0, 0.0)])[0].x
        assert slow_x < fast_x

    def test_converges_toward_a_held_value(self):
        smoother = LandmarkSmoother(alpha=0.5)
        smoother.smooth([(0.0, 0.0, 0.0)])
        for _ in range(20):
            result = smoother.smooth([(100.0, 0.0, 0.0)])
        assert result[0].x == pytest.approx(100.0, abs=0.01)

    def test_output_length_matches_input(self):
        frame = [(float(i), float(i), 0.0) for i in range(33)]
        assert len(LandmarkSmoother().smooth(frame)) == 33


class TestStateReset:
    def test_reset_reseeds_the_average(self):
        smoother = LandmarkSmoother(alpha=0.5)
        smoother.smooth([(0.0, 0.0, 0.0)])
        smoother.reset()
        assert smoother.smooth([(10.0, 0.0, 0.0)])[0].x == 10.0

    def test_landmark_count_change_reseeds_instead_of_blending(self):
        """A person leaving and re-entering must not blend mismatched points."""
        smoother = LandmarkSmoother(alpha=0.5)
        smoother.smooth([(0.0, 0.0, 0.0)])
        result = smoother.smooth([(10.0, 0.0, 0.0), (20.0, 0.0, 0.0)])
        assert [lm.x for lm in result] == [10.0, 20.0]

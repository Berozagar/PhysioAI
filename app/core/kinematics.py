"""
kinematics.py
=============

Folder location: PhysioAI/app/core/kinematics.py

Purpose
-------
Pure mathematical primitives used to turn raw landmark coordinates
(from any MediaPipe-compatible detector) into biomechanical
quantities used across the PhysioAI pipeline: distances, joint
angles, velocity, acceleration, smoothing, range of motion, and a
generic movement-smoothness proxy.

Design notes
------------
- This module has ZERO dependency on detector implementations
  (app/detectors/*) or exercise-specific logic (app/models/*). It only
  ever consumes plain (x, y) / (x, y, z) landmark tuples.
- Every function is stateless and side-effect free, so it can be unit
  tested in isolation and reused by MotionAnalyzer, exercise_rules.py,
  posture_rules.py, or any future module.
- No global variables are used. Constants are module-level but
  immutable (EPSILON, type aliases).
- This module targets PhysioAI's body-pose exercises (shoulder, elbow,
  knee, back, neck), not finger/hand tracking -- the project's only
  detector (app/detectors/pose_detector.py) is a 33-point MediaPipe
  Pose Landmarker, not a hand landmarker.

Author: Ramya (Core module)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Sequence, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
Point2D = Tuple[float, float]
Point3D = Tuple[float, float, float]
LandmarkPoint = Union[Point2D, Point3D]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Small value used to avoid division-by-zero in angle and smoothness
# calculations without introducing branching everywhere.
EPSILON: float = 1e-8

# ---------------------------------------------------------------------------
# Data classes (lightweight, immutable result containers)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ROMResult:
    """Result of a range-of-motion calculation over a sequence of angles."""
    min_angle: float
    max_angle: float
    range_of_motion: float


# ---------------------------------------------------------------------------
# Distance
# ---------------------------------------------------------------------------
def euclidean_distance(point_a: LandmarkPoint, point_b: LandmarkPoint) -> float:
    """
    Calculate the Euclidean (straight-line) distance between two landmarks.

    Mathematics
    -----------
    For points A and B in n-dimensional space:
        d(A, B) = sqrt( sum_i (A_i - B_i)^2 )
    Works transparently for 2D (x, y) or 3D (x, y, z) landmarks because
    the sum simply has more terms.

    Args:
        point_a: First landmark, e.g. (x1, y1, z1) or (x1, y1).
        point_b: Second landmark, same dimensionality as point_a.

    Returns:
        Non-negative float distance between the two points.

    Raises:
        ValueError: If point_a and point_b have different dimensionality.

    Example:
        >>> euclidean_distance((0, 0, 0), (3, 4, 0))
        5.0
    """
    if len(point_a) != len(point_b):
        raise ValueError(
            f"Landmark dimensionality mismatch: {len(point_a)} vs {len(point_b)}"
        )
    a = np.asarray(point_a, dtype=np.float64)
    b = np.asarray(point_b, dtype=np.float64)
    return float(np.linalg.norm(a - b))


# ---------------------------------------------------------------------------
# Angles
# ---------------------------------------------------------------------------
def calculate_angle(
    point_a: LandmarkPoint, point_b: LandmarkPoint, point_c: LandmarkPoint
) -> float:
    """
    Calculate the angle (degrees) at vertex `point_b`, formed by the rays
    point_b -> point_a and point_b -> point_c.

    Mathematics
    -----------
    Using the dot-product definition of the angle between two vectors:
        u = A - B,  v = C - B
        cos(theta) = (u . v) / (|u| * |v|)
        theta = arccos(cos(theta))
    This single primitive is reused by every joint-angle helper below
    (elbow, shoulder, knee, back, neck) -- only the semantic meaning of
    A, B, C changes.

    Args:
        point_a: First point defining the angle (e.g. shoulder).
        point_b: Vertex of the angle (e.g. elbow).
        point_c: Third point defining the angle (e.g. wrist).

    Returns:
        Angle in degrees, in the range [0, 180].

    Raises:
        ValueError: If point_a or point_c coincides with point_b
            (zero-length vector -> angle is undefined).

    Example:
        >>> calculate_angle((0, 1), (0, 0), (1, 0))
        90.0
    """
    a = np.asarray(point_a, dtype=np.float64)
    b = np.asarray(point_b, dtype=np.float64)
    c = np.asarray(point_c, dtype=np.float64)

    vector_ba = a - b
    vector_bc = c - b

    norm_ba = np.linalg.norm(vector_ba)
    norm_bc = np.linalg.norm(vector_bc)

    if norm_ba < EPSILON or norm_bc < EPSILON:
        raise ValueError(
            "Degenerate landmark configuration: vertex point coincides "
            "with point_a or point_c."
        )

    cosine_angle = np.dot(vector_ba, vector_bc) / (norm_ba * norm_bc)
    # Floating point error can push cosine slightly outside [-1, 1];
    # clip before arccos to avoid NaN.
    cosine_angle = float(np.clip(cosine_angle, -1.0, 1.0))
    angle_rad = np.arccos(cosine_angle)
    return float(np.degrees(angle_rad))


def elbow_angle(
    shoulder: LandmarkPoint, elbow: LandmarkPoint, wrist: LandmarkPoint
) -> float:
    """
    Calculate elbow flexion/extension angle.

    MediaPipe Pose reference: shoulder = 11/12, elbow = 13/14 (vertex),
    wrist = 15/16.

    Args:
        shoulder: Shoulder landmark.
        elbow: Elbow landmark (angle vertex).
        wrist: Wrist landmark.

    Returns:
        Elbow angle in degrees (~180 = fully extended arm).
    """
    return calculate_angle(shoulder, elbow, wrist)


def shoulder_angle(
    hip: LandmarkPoint, shoulder: LandmarkPoint, elbow: LandmarkPoint
) -> float:
    """
    Calculate shoulder flexion/abduction angle.

    MediaPipe Pose reference: hip = 23/24, shoulder = 11/12 (vertex),
    elbow = 13/14.

    Args:
        hip: Hip landmark.
        shoulder: Shoulder landmark (angle vertex).
        elbow: Elbow landmark.

    Returns:
        Shoulder angle in degrees. Small when the arm hangs at the
        side, large as the arm is raised (e.g. a shoulder-raise
        exercise).
    """
    return calculate_angle(hip, shoulder, elbow)


def knee_angle(
    hip: LandmarkPoint, knee: LandmarkPoint, ankle: LandmarkPoint
) -> float:
    """
    Calculate knee flexion/extension angle.

    MediaPipe Pose reference: hip = 23/24, knee = 25/26 (vertex),
    ankle = 27/28.

    Args:
        hip: Hip landmark.
        knee: Knee landmark (angle vertex).
        ankle: Ankle landmark.

    Returns:
        Knee angle in degrees (~180 = leg straight, smaller = bent,
        e.g. during a squat or knee-bend exercise).
    """
    return calculate_angle(hip, knee, ankle)


def back_angle(
    shoulder: LandmarkPoint, hip: LandmarkPoint, knee: LandmarkPoint
) -> float:
    """
    Calculate torso (back) alignment angle at the hip.

    Used for posture checks (e.g. "straighten your back" during a
    squat or shoulder raise) rather than for a specific exercise's
    primary movement.

    MediaPipe Pose reference: shoulder = 11/12, hip = 23/24 (vertex),
    knee = 25/26.

    Args:
        shoulder: Shoulder landmark.
        hip: Hip landmark (angle vertex).
        knee: Knee landmark.

    Returns:
        Back angle in degrees (~180 = upright/neutral spine, smaller
        = leaning/rounding forward at the hip).
    """
    return calculate_angle(shoulder, hip, knee)


def neck_angle(
    ear: LandmarkPoint, shoulder: LandmarkPoint, hip: LandmarkPoint
) -> float:
    """
    Calculate head/neck alignment angle at the shoulder.

    A simplified proxy for forward-head posture: the angle between the
    ear and the hip, measured at the shoulder. Close to 180 degrees
    means the ear is roughly stacked above the hip (neutral head
    position); a smaller angle indicates the head is drifting forward.

    MediaPipe Pose reference: ear = 7/8, shoulder = 11/12 (vertex),
    hip = 23/24.

    Args:
        ear: Ear landmark.
        shoulder: Shoulder landmark (angle vertex).
        hip: Hip landmark.

    Returns:
        Neck alignment angle in degrees.
    """
    return calculate_angle(ear, shoulder, hip)


# ---------------------------------------------------------------------------
# Velocity / acceleration
# ---------------------------------------------------------------------------
def calculate_velocity(
    previous_point: LandmarkPoint, current_point: LandmarkPoint, delta_time: float
) -> float:
    """
    Calculate instantaneous velocity of a landmark between two frames.

    Mathematics
    -----------
        velocity = distance(previous, current) / delta_time

    Args:
        previous_point: Landmark position in the previous frame.
        current_point: Landmark position in the current frame.
        delta_time: Time elapsed between frames, in seconds (must be > 0).

    Returns:
        Velocity in landmark-units per second.

    Raises:
        ValueError: If delta_time is not positive.

    Example:
        >>> calculate_velocity((0, 0), (0, 1), 0.5)
        2.0
    """
    if delta_time <= 0:
        raise ValueError("delta_time must be a positive, non-zero value.")
    distance = euclidean_distance(previous_point, current_point)
    return distance / delta_time


def calculate_acceleration(
    previous_velocity: float, current_velocity: float, delta_time: float
) -> float:
    """
    Calculate instantaneous acceleration from two consecutive velocity
    samples.

    Mathematics
    -----------
        acceleration = (v_current - v_previous) / delta_time

    Args:
        previous_velocity: Velocity at the previous frame.
        current_velocity: Velocity at the current frame.
        delta_time: Time elapsed between frames, in seconds (must be > 0).

    Returns:
        Acceleration in landmark-units per second^2. Can be negative
        (deceleration).

    Raises:
        ValueError: If delta_time is not positive.

    Example:
        >>> calculate_acceleration(1.0, 3.0, 0.5)
        4.0
    """
    if delta_time <= 0:
        raise ValueError("delta_time must be a positive, non-zero value.")
    return (current_velocity - previous_velocity) / delta_time


# ---------------------------------------------------------------------------
# Smoothing
# ---------------------------------------------------------------------------
def moving_average(values: Sequence[float], window_size: int = 5) -> float:
    """
    Real-time moving-average smoothing: returns the average of the most
    recent `window_size` samples.

    Mathematics
    -----------
        MA = (1 / N) * sum(last N values)

    Intended use: called once per frame on a short rolling buffer of
    raw coordinates/angles/velocities to reduce landmark jitter before
    further analysis.

    Args:
        values: Sequence of numeric samples, most recent last.
        window_size: Number of trailing samples to average (must be > 0).

    Returns:
        Smoothed float value. Returns 0.0 if `values` is empty.

    Raises:
        ValueError: If window_size is not positive.

    Example:
        >>> moving_average([1, 2, 3, 4, 5], window_size=3)
        4.0
    """
    if window_size <= 0:
        raise ValueError("window_size must be a positive integer.")
    if not values:
        return 0.0
    window = list(values)[-window_size:]
    return float(np.mean(window))


def smooth_series(values: Sequence[float], window_size: int = 5) -> List[float]:
    """
    Offline moving-average smoothing of an entire signal (e.g. after a
    repetition completes and the full angle trace is available).

    Mathematics
    -----------
    Same moving-average formula as `moving_average`, applied with a
    sliding window across the whole series:
        smoothed[i] = mean(values[max(0, i-w+1) : i+1])

    Args:
        values: Full sequence of numeric samples.
        window_size: Size of the smoothing window (must be > 0).

    Returns:
        A list the same length as `values`, smoothed.

    Raises:
        ValueError: If window_size is not positive.

    Example:
        >>> smooth_series([1, 2, 3, 4, 5], window_size=2)
        [1.0, 1.5, 2.5, 3.5, 4.5]
    """
    if window_size <= 0:
        raise ValueError("window_size must be a positive integer.")
    if not values:
        return []
    arr = np.asarray(values, dtype=np.float64)
    smoothed = [
        float(np.mean(arr[max(0, i - window_size + 1): i + 1])) for i in range(len(arr))
    ]
    return smoothed


# ---------------------------------------------------------------------------
# Range of motion
# ---------------------------------------------------------------------------
def calculate_range_of_motion(angle_sequence: Sequence[float]) -> ROMResult:
    """
    Calculate the range of motion (ROM) covered by a sequence of joint
    angles, typically spanning one repetition of an exercise.

    Mathematics
    -----------
        ROM = max(angles) - min(angles)

    Args:
        angle_sequence: Sequence of joint angles in degrees, must
            contain at least one value.

    Returns:
        ROMResult(min_angle, max_angle, range_of_motion).

    Raises:
        ValueError: If angle_sequence is empty.

    Example:
        >>> calculate_range_of_motion([30, 45, 90, 60])
        ROMResult(min_angle=30.0, max_angle=90.0, range_of_motion=60.0)
    """
    if not angle_sequence:
        raise ValueError("angle_sequence must contain at least one value.")
    min_angle = float(min(angle_sequence))
    max_angle = float(max(angle_sequence))
    return ROMResult(
        min_angle=min_angle,
        max_angle=max_angle,
        range_of_motion=max_angle - min_angle,
    )


# ---------------------------------------------------------------------------
# Movement smoothness
# ---------------------------------------------------------------------------
def calculate_smoothness(velocity_sequence: Sequence[float]) -> float:
    """
    Estimate a generic 0-100 "movement smoothness" score from a
    sequence of recent velocity samples.

    This is an internal exercise-performance proxy, NOT a clinical or
    diagnostic measurement. It rewards movement that proceeds at a
    consistent pace (low relative variability in velocity) over
    movement that is jerky or erratic.

    Mathematics
    -----------
    Uses the coefficient of variation (CV) of the velocity signal:
        cv = std(v) / (mean(v) + EPSILON)
        smoothness = 100 * (1 - min(cv, 1))
    A perfectly steady velocity (cv = 0) scores 100; highly erratic
    velocity (cv >= 1) scores 0.

    Args:
        velocity_sequence: Sequence of recent velocity magnitudes
            (landmark-units/second), same units throughout.

    Returns:
        Smoothness score in [0, 100]. Returns 0.0 if fewer than 2
        samples are available.

    Example:
        >>> round(calculate_smoothness([0.10, 0.11, 0.10, 0.09, 0.10]), 1)
        94.9
    """
    if len(velocity_sequence) < 2:
        return 0.0
    arr = np.asarray(velocity_sequence, dtype=np.float64)
    mean_v = float(np.mean(arr))
    std_v = float(np.std(arr))
    cv = std_v / (mean_v + EPSILON)
    score = 100.0 * (1.0 - min(cv, 1.0))
    return float(np.clip(score, 0.0, 100.0))
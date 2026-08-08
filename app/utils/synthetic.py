"""
===========================================
Synthetic Pose Generator
-------------------------------------------
Builds 33-point MediaPipe Pose-shaped landmark
lists from a target joint angle, with no camera
and no model involved.
===========================================

Used by `run.py --self-test` to drive the full analysis pipeline
(MotionAnalyzer -> ExerciseRule -> PostureRules -> Drawer -> Audio)
on machines with no webcam, and to sanity-check rep counting against
a known-good angle sweep.

The body is drawn in normalized image coordinates (0.0-1.0, y growing
downwards, exactly like MediaPipe) on the LEFT side of the body, which
is the side MotionAnalyzer's DEFAULT_JOINT_TRIPLETS reads.
"""

from __future__ import annotations

import math
from typing import List, Tuple

Point3D = Tuple[float, float, float]

NUM_POSE_LANDMARKS: int = 33

# MediaPipe Pose indices this generator positions meaningfully.
NOSE = 0
LEFT_EAR = 7
LEFT_SHOULDER = 11
LEFT_ELBOW = 13
LEFT_WRIST = 15
LEFT_HIP = 23
LEFT_KNEE = 25
LEFT_ANKLE = 27

# Skeleton proportions in normalized units.
_SHOULDER = (0.50, 0.40)
_HIP = (0.50, 0.60)
_UPPER_ARM = 0.16
_FOREARM = 0.16
_THIGH = 0.18
_SHIN = 0.18


def create_pose_landmarks(
    shoulder_angle_deg: float,
    knee_angle_deg: float = 180.0,
    nose_y: float = 0.20,
) -> List[Point3D]:
    """
    Build a full 33-landmark pose with the requested joint angles.

    Args:
        shoulder_angle_deg: Desired angle at the shoulder, measured
            between the hip and the elbow. 0 = arm hanging straight
            down, 180 = arm straight overhead.
        knee_angle_deg: Desired angle at the knee, measured between
            the hip and the ankle. 180 = leg straight.
        nose_y: Vertical position of the nose, used to inject overall
            body movement between frames so velocity is non-zero.

    Returns:
        A list of exactly 33 (x, y, z) tuples, the same shape
        PoseDetector.get_landmarks() returns.
    """
    landmarks: List[Point3D] = [(0.5, 0.5, 0.0)] * NUM_POSE_LANDMARKS

    shoulder_x, shoulder_y = _SHOULDER
    hip_x, hip_y = _HIP

    # The shoulder angle is measured at the shoulder between the ray to
    # the hip (straight down) and the ray to the elbow. Rotating the
    # "straight down" unit vector (0, 1) by theta gives (sin, cos).
    theta = math.radians(shoulder_angle_deg)
    arm_dx, arm_dy = math.sin(theta), math.cos(theta)

    elbow_x = shoulder_x + _UPPER_ARM * arm_dx
    elbow_y = shoulder_y + _UPPER_ARM * arm_dy
    # Straight arm: the forearm continues along the same direction, so
    # the elbow angle stays at 180 degrees.
    wrist_x = elbow_x + _FOREARM * arm_dx
    wrist_y = elbow_y + _FOREARM * arm_dy

    knee_x = hip_x
    knee_y = hip_y + _THIGH

    # Same construction at the knee: rotate the downward ray from the
    # knee by (180 - knee_angle) so 180 degrees keeps the leg straight.
    phi = math.radians(180.0 - knee_angle_deg)
    ankle_x = knee_x + _SHIN * math.sin(phi)
    ankle_y = knee_y + _SHIN * math.cos(phi)

    landmarks[NOSE] = (shoulder_x, nose_y, 0.0)
    landmarks[LEFT_EAR] = (shoulder_x, shoulder_y - 0.10, 0.0)
    landmarks[LEFT_SHOULDER] = (shoulder_x, shoulder_y, 0.0)
    landmarks[LEFT_ELBOW] = (elbow_x, elbow_y, 0.0)
    landmarks[LEFT_WRIST] = (wrist_x, wrist_y, 0.0)
    landmarks[LEFT_HIP] = (hip_x, hip_y, 0.0)
    landmarks[LEFT_KNEE] = (knee_x, knee_y, 0.0)
    landmarks[LEFT_ANKLE] = (ankle_x, ankle_y, 0.0)

    return landmarks


# A real repetition ramps over roughly half a second and pauses briefly at
# each end. Sweeping faster than this is not just unrealistic -- it outruns
# LandmarkSmoother's exponential moving average, which lags a ramp by
# (1 - alpha) / alpha frames and therefore clips the peak angle the
# exercise rule is waiting for.
DEFAULT_STEPS_PER_PHASE: int = 12
DEFAULT_HOLD_FRAMES: int = 8

# How far past each of a rule's thresholds the synthetic body travels. A real
# user overshoots the threshold rather than stopping exactly on it, and the
# margin also absorbs the smoothing lag, which would otherwise leave the
# measured angle just short of the threshold and never complete the rep.
DEFAULT_MARGIN_DEGREES: float = 15.0

# Anatomical limits of the joint angles this generator can express.
MIN_ANGLE: float = 10.0
MAX_ANGLE: float = 180.0


def _sweep(
    start: float,
    target: float,
    repetitions: int,
    steps_per_phase: int,
    hold_frames: int,
) -> List[float]:
    """
    Build a start -> target -> start angle sweep with a pause at each end.

    Args:
        start: Resting angle each cycle begins and ends at.
        target: Peak angle reached mid-cycle.
        repetitions: Number of complete cycles.
        steps_per_phase: Frames spent ramping in each direction.
        hold_frames: Frames held still at each extreme.

    Returns:
        List of angles in degrees.
    """
    span = target - start
    ramp = [
        start + span * (i / (steps_per_phase - 1)) for i in range(steps_per_phase)
    ]

    angles: List[float] = []
    for _ in range(repetitions):
        angles.extend(ramp)
        angles.extend([target] * hold_frames)
        angles.extend(list(reversed(ramp))[1:])
        angles.extend([start] * hold_frames)

    return angles


def cycle_for_rule(
    start_angle: float,
    end_angle: float,
    repetitions: int = 2,
    steps_per_phase: int = DEFAULT_STEPS_PER_PHASE,
    hold_frames: int = DEFAULT_HOLD_FRAMES,
    margin: float = DEFAULT_MARGIN_DEGREES,
) -> List[float]:
    """
    Build an angle sweep that completes `repetitions` reps of any exercise
    rule, derived from that rule's own thresholds.

    The sweep travels `margin` degrees beyond both `start_angle` and
    `end_angle` (clamped to anatomically valid angles), so it works for
    any exercise added to EXERCISES without hand-tuned constants.

    Args:
        start_angle: The rule's resting-position threshold.
        end_angle: The rule's completed-position threshold.
        repetitions: How many full cycles to generate.
        steps_per_phase: Frames spent ramping in each direction.
        hold_frames: Frames held still at each extreme.
        margin: Degrees of overshoot past each threshold.

    Returns:
        List of joint angles in degrees.
    """
    # +1 when the angle grows through the rep (shoulder raise), -1 when it
    # shrinks (squat, knee bend).
    direction = 1.0 if end_angle > start_angle else -1.0

    rest = _clamp(start_angle - direction * margin)
    peak = _clamp(end_angle + direction * margin)

    return _sweep(rest, peak, repetitions, steps_per_phase, hold_frames)


def _clamp(angle: float) -> float:
    """Keep a generated angle within the range this skeleton can express."""
    return max(MIN_ANGLE, min(MAX_ANGLE, angle))


def shoulder_raise_cycle(
    repetitions: int = 2,
    steps_per_phase: int = DEFAULT_STEPS_PER_PHASE,
    hold_frames: int = DEFAULT_HOLD_FRAMES,
) -> List[float]:
    """
    Produce a shoulder-raise sweep crossing the shoulder_raise rule's
    start_angle (30) and end_angle (160) in both directions.

    Args:
        repetitions: How many up-down cycles to generate.
        steps_per_phase: Frames spent ramping in each direction.
        hold_frames: Frames held still at the top and bottom.

    Returns:
        List of shoulder angles in degrees.
    """
    return cycle_for_rule(30.0, 160.0, repetitions, steps_per_phase, hold_frames)


def squat_cycle(
    repetitions: int = 2,
    steps_per_phase: int = DEFAULT_STEPS_PER_PHASE,
    hold_frames: int = DEFAULT_HOLD_FRAMES,
) -> List[float]:
    """
    Produce a squat knee-angle sweep crossing the squat rule's
    start_angle (170) and end_angle (90).

    Args:
        repetitions: How many down-up cycles to generate.
        steps_per_phase: Frames spent ramping in each direction.
        hold_frames: Frames held still at the bottom and top.

    Returns:
        List of knee angles in degrees.
    """
    return cycle_for_rule(170.0, 90.0, repetitions, steps_per_phase, hold_frames)

"""
motion_analyzer.py
===================

Folder location: PhysioAI/app/core/motion_analyzer.py

Purpose
-------
Consumes per-frame landmark sequences (produced by app/detectors/*,
in MediaPipe-compatible format) and turns them into a rolling
kinematic analysis: joint angles, velocity, acceleration, movement
detection, pause detection, range of motion, and a generic movement
smoothness/quality signal.

Design notes
------------
- DETECTOR-AGNOSTIC: this module never imports pose_detector.py. It
  only assumes landmarks arrive as either plain (x, y, z) tuples OR
  MediaPipe NormalizedLandmark-like objects (anything exposing
  .x / .y / .z attributes). `_normalize_landmarks` is the single seam
  where that flexibility lives.
- EXERCISE-AGNOSTIC BY DEFAULT: there is no hardcoded exercise logic
  (e.g. "squat", "shoulder raise") anywhere in this file. Which joints
  to track is supplied via `joint_config`, a mapping of
  joint_name -> landmark index triplet. The defaults below match the
  joint names PhysioAI's exercise/posture rules already use
  (shoulder, elbow, knee, back, neck) against the 33-point MediaPipe
  Pose landmark order used by app/detectors/pose_detector.py.
- RULE INTEGRATION: repetition counting, exercise stage, progress, and
  exercise-specific feedback text are already implemented in
  app/models/exercise_rules.py (ExerciseRule). This module does not
  duplicate that state machine. Instead, `analyze_frame` accepts an
  optional `exercise_rule` (any object exposing `.rule["joint"]` and
  `.get_status(angle) -> dict`, matching ExerciseRule's interface) and
  delegates repetition/stage/progress/feedback to it for the
  configured primary joint. When no exercise_rule is supplied (e.g.
  standalone testing, as in test_core.py), this module falls back to
  a generic, exercise-agnostic repetition counter and generic
  feedback strings so it remains independently usable/testable.
- STATE: frame history is kept in bounded `collections.deque` buffers
  (no unbounded growth, no global variables). Each analyzer instance
  owns its own state, so multiple MotionAnalyzer instances (e.g. one
  per patient session) can run independently.

Author: Ramya (Core module)
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple, Union

from app.core.kinematics import (
    ROMResult,
    calculate_acceleration as kinematics_calculate_acceleration,
    calculate_angle,
    calculate_range_of_motion,
    calculate_smoothness,
    calculate_velocity as kinematics_calculate_velocity,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
Point3D = Tuple[float, float, float]
LandmarkList = List[Point3D]
RawLandmark = Union[Tuple[float, float], Point3D, Any]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_HISTORY_LENGTH: int = 60          # ~2 seconds of buffer at 30 FPS
DEFAULT_REFERENCE_LANDMARK_INDEX: int = 0  # NOSE (MediaPipe Pose) -- overall body movement proxy
DEFAULT_MOTION_THRESHOLD: float = 0.02     # landmark-units/sec to count as "moving"
DEFAULT_REP_ANGLE_THRESHOLD: float = 15.0  # degrees of change to register a direction flip
DEFAULT_MIN_DIRECTION_CONFIRM_FRAMES: int = 2  # consecutive frames needed to confirm a reversal
DEFAULT_PAUSE_VELOCITY_THRESHOLD: float = 0.01
DEFAULT_PAUSE_DURATION_THRESHOLD: float = 0.5  # seconds

# Generic default joint config, matching MediaPipe Pose's fixed
# 33-landmark order (see app/detectors/pose_detector.py) and the
# joint-name vocabulary already used by app/models/exercise_rules.py
# and app/models/posture_rules.py ("shoulder", "knee", "back", "neck").
# Uses the LEFT side of the body by default; pass a custom
# AnalyzerConfig.joint_config for the right side or a specific
# exercise's landmark set.
#   7  = LEFT_EAR        11 = LEFT_SHOULDER   13 = LEFT_ELBOW
#   15 = LEFT_WRIST       23 = LEFT_HIP        25 = LEFT_KNEE
#   27 = LEFT_ANKLE
DEFAULT_JOINT_TRIPLETS: Dict[str, Tuple[int, int, int]] = {
    "shoulder": (23, 11, 13),  # hip, shoulder (vertex), elbow
    "elbow": (11, 13, 15),     # shoulder, elbow (vertex), wrist
    "knee": (23, 25, 27),      # hip, knee (vertex), ankle
    "back": (11, 23, 25),      # shoulder, hip (vertex), knee
    "neck": (7, 11, 23),       # ear, shoulder (vertex), hip
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class FrameRecord:
    """Snapshot of one analyzed frame, kept in the rolling history buffer."""
    landmarks: LandmarkList
    timestamp: float


@dataclass
class AnalyzerConfig:
    """Tunable parameters for a MotionAnalyzer instance."""
    joint_config: Dict[str, Tuple[int, int, int]] = field(
        default_factory=lambda: dict(DEFAULT_JOINT_TRIPLETS)
    )
    history_length: int = DEFAULT_HISTORY_LENGTH
    reference_landmark_index: int = DEFAULT_REFERENCE_LANDMARK_INDEX
    motion_threshold: float = DEFAULT_MOTION_THRESHOLD
    rep_angle_threshold: float = DEFAULT_REP_ANGLE_THRESHOLD
    min_direction_confirm_frames: int = DEFAULT_MIN_DIRECTION_CONFIRM_FRAMES
    pause_velocity_threshold: float = DEFAULT_PAUSE_VELOCITY_THRESHOLD
    pause_duration_threshold: float = DEFAULT_PAUSE_DURATION_THRESHOLD


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------
class MotionAnalyzer:
    """
    Analyzes a live stream of body-pose landmarks frame by frame.

    Typical usage (standalone, generic):

        analyzer = MotionAnalyzer()
        while capturing:
            landmarks = detector.get_landmarks(frame)  # from PoseDetector
            result = analyzer.analyze_frame(landmarks)
            print(result["movement_quality"], result["feedback"])

    Typical usage (integrated with an exercise's rules):

        from app.models.exercise_rules import ExerciseRule

        analyzer = MotionAnalyzer()
        rule = ExerciseRule("shoulder_raise")
        while capturing:
            landmarks = detector.get_landmarks(frame)
            result = analyzer.analyze_frame(landmarks, exercise_rule=rule)
            print(result["repetitions"], result["feedback"])

    The class is deliberately generic: by default it has no idea what
    exercise is being performed. `joint_config` tells it which
    three-landmark triplets define the joints to track; velocity,
    acceleration, ROM, and smoothness all fall out of the kinematics
    primitives. Repetition counting and exercise-specific feedback are
    generic fallbacks unless an `exercise_rule` is passed to
    `analyze_frame`, in which case that rule's own thresholds/stage
    machine (app/models/exercise_rules.py) take over.
    """

    def __init__(self, config: Optional[AnalyzerConfig] = None) -> None:
        """
        Args:
            config: Optional AnalyzerConfig. If omitted, sensible
                generic defaults are used (see AnalyzerConfig).
        """
        self._config = config or AnalyzerConfig()

        # Rolling history buffers -- bounded, instance-owned (no globals).
        self._landmark_history: Deque[FrameRecord] = deque(
            maxlen=self._config.history_length
        )
        self._velocity_history: Deque[float] = deque(maxlen=self._config.history_length)
        self._joint_angle_histories: Dict[str, Deque[float]] = {}

        # Repetition-counting state (generic fallback path only).
        self._rep_count: int = 0
        self._rep_direction: Optional[str] = None  # "increasing" | "decreasing"
        self._last_extremum_angle: Optional[float] = None
        self._pending_direction: Optional[str] = None
        self._pending_frames: int = 0

        # Pause-detection state.
        self._pause_start_time: Optional[float] = None

        logger.debug("MotionAnalyzer initialized with config: %s", self._config)

    # ------------------------------------------------------------------
    # Landmark normalization (the detector-decoupling seam)
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_landmarks(raw_landmarks: Sequence[RawLandmark]) -> LandmarkList:
        """
        Convert detector output into a plain list of (x, y, z) tuples,
        regardless of whether it arrives as tuples/lists or as
        MediaPipe NormalizedLandmark-like objects (duck-typed via
        hasattr, so no hard dependency on the mediapipe package here).

        Args:
            raw_landmarks: Sequence of landmarks in either
                (x, y) / (x, y, z) tuple form, or objects exposing
                .x / .y / (optionally) .z attributes.

        Returns:
            List of (x, y, z) float tuples, z defaults to 0.0 when
            the source landmark is 2D.

        Raises:
            TypeError: If an element is neither tuple/list nor an
                object exposing .x/.y.
            ValueError: If a tuple/list landmark has fewer than 2
                coordinates.
        """
        normalized: LandmarkList = []
        for lm in raw_landmarks:
            if hasattr(lm, "x") and hasattr(lm, "y"):
                z = getattr(lm, "z", 0.0)
                normalized.append((float(lm.x), float(lm.y), float(z)))
            elif isinstance(lm, (tuple, list)):
                if len(lm) == 2:
                    normalized.append((float(lm[0]), float(lm[1]), 0.0))
                elif len(lm) >= 3:
                    normalized.append((float(lm[0]), float(lm[1]), float(lm[2])))
                else:
                    raise ValueError(f"Landmark tuple has invalid length: {lm}")
            else:
                raise TypeError(f"Unsupported landmark type: {type(lm)!r}")
        return normalized

    # ------------------------------------------------------------------
    # Per-metric calculations
    # ------------------------------------------------------------------
    def calculate_joint_angles(self, landmarks: LandmarkList) -> Dict[str, float]:
        """
        Compute every configured joint angle for the current frame.

        Args:
            landmarks: Normalized (x, y, z) landmark list for this frame.

        Returns:
            Dict mapping joint_name -> angle in degrees. Joints whose
            required landmark indices are unavailable, or whose points
            are degenerate, are silently skipped (and logged).
        """
        angles: Dict[str, float] = {}
        for joint_name, (a_idx, b_idx, c_idx) in self._config.joint_config.items():
            if max(a_idx, b_idx, c_idx) >= len(landmarks):
                logger.debug(
                    "Skipping joint '%s': only %d landmarks available.",
                    joint_name, len(landmarks),
                )
                continue
            try:
                angles[joint_name] = calculate_angle(
                    landmarks[a_idx], landmarks[b_idx], landmarks[c_idx]
                )
            except ValueError as exc:
                logger.warning("Angle calculation failed for '%s': %s", joint_name, exc)
        return angles

    def calculate_velocity(self, landmarks: LandmarkList, timestamp: float) -> float:
        """
        Compute the velocity of the reference landmark (default: nose,
        index 0, as a general body-movement proxy) between the
        previous stored frame and this one.

        Args:
            landmarks: Normalized landmark list for the current frame.
            timestamp: Current frame timestamp, in seconds.

        Returns:
            Velocity in landmark-units/second. 0.0 if there is no
            previous frame yet, or if dt <= 0 (e.g. duplicate timestamp).
        """
        if not self._landmark_history:
            return 0.0
        previous = self._landmark_history[-1]
        dt = timestamp - previous.timestamp
        ref_idx = self._config.reference_landmark_index
        if dt <= 0 or ref_idx >= len(landmarks) or ref_idx >= len(previous.landmarks):
            return 0.0
        return kinematics_calculate_velocity(
            previous.landmarks[ref_idx], landmarks[ref_idx], dt
        )

    def calculate_acceleration(self, current_velocity: float, timestamp: float) -> float:
        """
        Compute acceleration from the previous stored velocity/timestamp
        to the current velocity.

        Args:
            current_velocity: Velocity computed for the current frame.
            timestamp: Current frame timestamp, in seconds.

        Returns:
            Acceleration in landmark-units/second^2. 0.0 if there is no
            prior sample yet, or if dt <= 0.
        """
        if not self._velocity_history or not self._landmark_history:
            return 0.0
        dt = timestamp - self._landmark_history[-1].timestamp
        if dt <= 0:
            return 0.0
        previous_velocity = self._velocity_history[-1]
        return kinematics_calculate_acceleration(previous_velocity, current_velocity, dt)

    def detect_motion(self, velocity: float) -> bool:
        """
        Decide whether the tracked point is currently "in motion" based
        on a simple velocity threshold.

        Args:
            velocity: Velocity computed for the current frame.

        Returns:
            True if velocity exceeds the configured motion_threshold.
        """
        return velocity > self._config.motion_threshold

    def calculate_smoothness(self) -> float:
        """
        Estimate movement smoothness from the recent velocity history
        buffer (see kinematics.calculate_smoothness). This is a
        generic exercise-performance proxy, not a clinical measure.

        Returns:
            Smoothness score in [0, 100]. 0.0 if fewer than 2 velocity
            samples are buffered yet.
        """
        if len(self._velocity_history) < 2:
            return 0.0
        return calculate_smoothness(list(self._velocity_history))

    def evaluate_range_of_motion(self, joint_name: str) -> Optional[ROMResult]:
        """
        Compute the range of motion for a given joint over its buffered
        angle history.

        Args:
            joint_name: Name of a joint present in `joint_config`.

        Returns:
            ROMResult, or None if the joint has no buffered history yet.
        """
        history = self._joint_angle_histories.get(joint_name)
        if not history:
            return None
        return calculate_range_of_motion(list(history))

    def count_repetitions(self, primary_angle: float) -> int:
        """
        Generic, exercise-agnostic repetition counter based on
        confirmed direction changes of a "primary" joint angle signal.

        A candidate direction change must persist for at least
        `min_direction_confirm_frames` consecutive qualifying frames
        (each exceeding `rep_angle_threshold` degrees of movement from
        the last confirmed extremum) before it is accepted as a real
        reversal and a repetition is counted. This dead-band +
        confirmation-frame combination avoids miscounting a single
        noisy frame as a full repetition, which a plain single-frame
        direction flip does not guard against.

        This is a fallback used only when no `exercise_rule` is passed
        to `analyze_frame` -- when a rule is provided, its own
        state machine (app/models/exercise_rules.py) is authoritative.

        Args:
            primary_angle: The current value of the joint angle being
                used to count repetitions.

        Returns:
            The updated total repetition count.
        """
        if self._last_extremum_angle is None:
            self._last_extremum_angle = primary_angle
            return self._rep_count

        delta = primary_angle - self._last_extremum_angle
        if abs(delta) < self._config.rep_angle_threshold:
            return self._rep_count  # not enough movement to be a real direction change

        candidate_direction = "increasing" if delta > 0 else "decreasing"
        if candidate_direction == self._pending_direction:
            self._pending_frames += 1
        else:
            self._pending_direction = candidate_direction
            self._pending_frames = 1

        if self._pending_frames >= self._config.min_direction_confirm_frames:
            if self._rep_direction is not None and candidate_direction != self._rep_direction:
                self._rep_count += 1
                logger.debug("Repetition counted. Total: %d", self._rep_count)
            self._rep_direction = candidate_direction
            self._last_extremum_angle = primary_angle
            self._pending_frames = 0

        return self._rep_count

    def detect_pauses(self, velocity: float, timestamp: float) -> bool:
        """
        Detect a sustained pause in movement (velocity below threshold
        for longer than pause_duration_threshold).

        Args:
            velocity: Velocity computed for the current frame.
            timestamp: Current frame timestamp, in seconds.

        Returns:
            True once the pause has lasted at least
            pause_duration_threshold seconds; False otherwise.
        """
        is_below_threshold = velocity < self._config.pause_velocity_threshold
        if not is_below_threshold:
            self._pause_start_time = None
            return False

        if self._pause_start_time is None:
            self._pause_start_time = timestamp
        elapsed = timestamp - self._pause_start_time
        return elapsed >= self._config.pause_duration_threshold

    def calculate_movement_quality(self, smoothness: float, paused: bool) -> float:
        """
        Combine smoothness with pause state into an overall 0-100
        generic movement-quality proxy.

        This is an internal exercise-performance signal, not a
        clinically validated score. It intentionally does NOT know
        about exercise-specific ROM targets -- that judgment belongs
        to app/models/exercise_rules.py, which has the actual target
        angles for the exercise being performed.

        Args:
            smoothness: Smoothness score for the current window (see
                calculate_smoothness).
            paused: Whether a sustained pause was detected this frame.

        Returns:
            Movement-quality score in [0, 100].
        """
        score = smoothness
        if paused:
            score *= 0.5
        return float(max(0.0, min(100.0, score)))

    def generate_feedback(
        self,
        movement_quality: float,
        paused: bool,
        in_motion: bool,
    ) -> str:
        """
        Produce a short, generic human-readable feedback string from
        the current metrics. Deliberately exercise-agnostic -- talks
        about movement quality in general terms only. This is only
        used when no `exercise_rule` is supplied to `analyze_frame`;
        exercise-specific phrasing (e.g. "raise your arm higher")
        comes from app/models/exercise_rules.py / posture_rules.py.

        Args:
            movement_quality: Score from calculate_movement_quality.
            paused: Whether a sustained pause was detected.
            in_motion: Whether the tracked point is currently moving.

        Returns:
            Short feedback sentence.
        """
        if paused:
            return "Exercise paused. Resume when ready."
        if not in_motion:
            return "Waiting for movement."
        if movement_quality >= 80:
            return "Good movement -- smooth and controlled."
        if movement_quality >= 50:
            return "Movement detected. Try to keep it steady."
        return "Move more slowly and with better control."

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def analyze_frame(
        self,
        raw_landmarks: Sequence[RawLandmark],
        timestamp: Optional[float] = None,
        exercise_rule: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Analyze a single frame of landmarks and update internal history.

        Args:
            raw_landmarks: Landmarks for this frame, either as
                (x, y[, z]) tuples or MediaPipe-style landmark objects.
            timestamp: Frame timestamp in seconds. Defaults to
                time.time() if not supplied (e.g. when the detector
                does not provide capture timestamps).
            exercise_rule: Optional exercise rule object (e.g. an
                app.models.exercise_rules.ExerciseRule instance).
                When supplied, its target joint (`exercise_rule.rule
                ["joint"]`) is used to select the primary joint angle,
                and its `.get_status(angle)` result supplies
                repetitions, stage, progress, and feedback -- this
                module does not invent exercise-specific feedback
                itself. When omitted, a generic fallback is used.

        Returns:
            Dictionary with keys: "angles", "velocity", "acceleration",
            "smoothness", "repetitions", "movement_quality",
            "range_of_motion", "in_motion", "paused", "feedback",
            "stage", "progress", "exercise". "stage"/"progress"/
            "exercise" are None unless an exercise_rule was supplied.
            On internal error, a safe zeroed-out dictionary plus an
            "error" key is returned instead of raising, so a single
            bad frame never crashes the video loop.
        """
        try:
            ts = timestamp if timestamp is not None else time.time()
            landmarks = self._normalize_landmarks(raw_landmarks)

            angles = self.calculate_joint_angles(landmarks)
            velocity = self.calculate_velocity(landmarks, ts)
            acceleration = self.calculate_acceleration(velocity, ts)
            in_motion = self.detect_motion(velocity)
            paused = self.detect_pauses(velocity, ts)

            # Update rolling history buffers.
            self._velocity_history.append(velocity)
            for joint_name, angle_value in angles.items():
                self._joint_angle_histories.setdefault(
                    joint_name, deque(maxlen=self._config.history_length)
                ).append(angle_value)
            self._landmark_history.append(FrameRecord(landmarks=landmarks, timestamp=ts))

            smoothness = self.calculate_smoothness()

            # Choose the "primary" joint: the exercise rule's target
            # joint if one was supplied and available this frame,
            # otherwise the first configured joint that produced an
            # angle this frame.
            if exercise_rule is not None and exercise_rule.rule.get("joint") in angles:
                primary_joint = exercise_rule.rule["joint"]
            else:
                primary_joint = next(iter(angles), None)

            rom = self.evaluate_range_of_motion(primary_joint) if primary_joint else None
            movement_quality = self.calculate_movement_quality(smoothness, paused)

            stage: Optional[str] = None
            progress: Optional[float] = None
            exercise_name: Optional[str] = None

            if exercise_rule is not None and primary_joint in angles:
                status = exercise_rule.get_status(angles[primary_joint])
                repetitions = status["counter"]
                stage = status["stage"]
                progress = status["progress"]
                feedback = status["feedback"]
                exercise_name = status["exercise"]
            else:
                if primary_joint is not None:
                    repetitions = self.count_repetitions(angles[primary_joint])
                else:
                    repetitions = self._rep_count
                feedback = self.generate_feedback(movement_quality, paused, in_motion)

            return {
                "angles": angles,
                "velocity": velocity,
                "acceleration": acceleration,
                "smoothness": smoothness,
                "repetitions": repetitions,
                "movement_quality": movement_quality,
                "range_of_motion": rom.range_of_motion if rom is not None else 0.0,
                "in_motion": in_motion,
                "paused": paused,
                "feedback": feedback,
                "stage": stage,
                "progress": progress,
                "exercise": exercise_name,
            }
        except Exception as exc:  # noqa: BLE001 - deliberate top-level safety net
            logger.exception("analyze_frame failed: %s", exc)
            return {
                "angles": {},
                "velocity": 0.0,
                "acceleration": 0.0,
                "smoothness": 0.0,
                "repetitions": self._rep_count,
                "movement_quality": 0.0,
                "range_of_motion": 0.0,
                "in_motion": False,
                "paused": False,
                "feedback": "Analysis error.",
                "stage": None,
                "progress": None,
                "exercise": None,
                "error": str(exc),
            }

    # ------------------------------------------------------------------
    # Session control
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """
        Clear all rolling history and counters. Call this at the start
        of a new exercise session/repetition set so metrics from a
        previous session don't leak into the new one. Note: this
        resets MotionAnalyzer's own generic fallback rep counter only
        -- if you are using an `exercise_rule`, also call its own
        `.reset()` method.
        """
        self._landmark_history.clear()
        self._velocity_history.clear()
        self._joint_angle_histories.clear()
        self._rep_count = 0
        self._rep_direction = None
        self._last_extremum_angle = None
        self._pending_direction = None
        self._pending_frames = 0
        self._pause_start_time = None
        logger.info("MotionAnalyzer state reset.")
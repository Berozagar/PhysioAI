"""
motion_analyzer.py
===================

Folder location: NeuroHand/app/core/motion_analyzer.py

Purpose
-------
Consumes per-frame landmark sequences (produced by app/detectors/*,
in MediaPipe-compatible format) and turns them into a rolling
kinematic analysis: joint angles, velocity, acceleration, tremor,
repetitions, range of motion, and an overall motion quality score.

Design notes
------------
- DETECTOR-AGNOSTIC: this module never imports hand_detector.py or
  pose_detector.py. It only assumes landmarks arrive as either plain
  (x, y, z) tuples OR MediaPipe NormalizedLandmark-like objects
  (anything exposing .x / .y / .z attributes). `_normalize_landmarks`
  is the single seam where that flexibility lives.
- EXERCISE-AGNOSTIC: there is noapp hardcoded exercise logic (e.g. "bicep
  curl", "wrist flexion") anywhere in this file. Which joints to track
  is supplied via `joint_config`, a mapping of joint_name -> landmark
  index triplet. Once app/models/exercise_rules.py exists, it can
  inject exercise-specific joint configs and thresholds into this
  class without any changes here.
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
    calculate_stability_score,
    calculate_velocity as kinematics_calculate_velocity,
    estimate_tremor,
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
DEFAULT_REFERENCE_LANDMARK_INDEX: int = 0  # wrist landmark, used for overall velocity
DEFAULT_MOTION_THRESHOLD: float = 0.02     # landmark-units/sec to count as "moving"
DEFAULT_REP_ANGLE_THRESHOLD: float = 15.0  # degrees of change to register a direction flip
DEFAULT_PAUSE_VELOCITY_THRESHOLD: float = 0.01
DEFAULT_PAUSE_DURATION_THRESHOLD: float = 0.5  # seconds

# Generic placeholder joint config. Real, exercise-specific joint
# triplets are expected to come from app/models/exercise_rules.py;
# these defaults just keep the analyzer usable/testable standalone.
DEFAULT_JOINT_TRIPLETS: Dict[str, Tuple[int, int, int]] = {
    "index_finger": (5, 6, 8),
    "middle_finger": (9, 10, 12),
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
    pause_velocity_threshold: float = DEFAULT_PAUSE_VELOCITY_THRESHOLD
    pause_duration_threshold: float = DEFAULT_PAUSE_DURATION_THRESHOLD


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------
class MotionAnalyzer:
    """
    Analyzes a live stream of hand/pose landmarks frame by frame.

    Typical usage:

        analyzer = MotionAnalyzer()
        while capturing:
            landmarks = detector.get_landmarks(frame)  # from Avanti's module
            result = analyzer.analyze_frame(landmarks)
            print(result["motion_score"], result["feedback"])

    The class is deliberately generic: it has no idea what exercise is
    being performed. `joint_config` tells it which three-landmark
    triplets define the joints to track; everything else (velocity,
    tremor, reps, ROM, score) falls out of the kinematics primitives.
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

        # Repetition-counting state.
        self._rep_count: int = 0
        self._rep_direction: Optional[str] = None  # "increasing" | "decreasing"
        self._last_extremum_angle: Optional[float] = None

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
        Compute the velocity of the reference landmark (default: wrist,
        index 0) between the previous stored frame and this one.

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

    def detect_tremor(self) -> float:
        """
        Estimate tremor from the recent velocity history buffer.

        Returns:
            Tremor index (see kinematics.estimate_tremor). 0.0 if
            fewer than 2 velocity samples are buffered yet.
        """
        if len(self._velocity_history) < 2:
            return 0.0
        return estimate_tremor(list(self._velocity_history))

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
        Generic repetition counter based on direction changes of a
        "primary" joint angle signal (peak/trough counting).

        A repetition is registered every time the tracked angle changes
        direction (increasing -> decreasing, or vice versa) by more
        than `rep_angle_threshold` degrees since the last recorded
        extremum. This makes no assumption about which exercise is
        being performed -- the caller decides which joint is "primary".

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

        new_direction = "increasing" if delta > 0 else "decreasing"
        if self._rep_direction is not None and new_direction != self._rep_direction:
            self._rep_count += 1
            logger.debug("Repetition counted. Total: %d", self._rep_count)
        self._rep_direction = new_direction
        self._last_extremum_angle = primary_angle
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

    def calculate_motion_score(
        self, tremor: float, rom: Optional[ROMResult]
    ) -> float:
        """
        Combine tremor and range of motion into an overall 0-100 motion
        quality score (thin wrapper over kinematics.calculate_stability_score).

        Args:
            tremor: Tremor index for the current window.
            rom: ROMResult for the primary joint, or None if unavailable.

        Returns:
            Motion score in [0, 100].
        """
        rom_value = rom.range_of_motion if rom is not None else 0.0
        return calculate_stability_score(tremor, rom_value)

    def generate_analysis(
        self,
        motion_score: float,
        tremor: float,
        paused: bool,
        in_motion: bool,
    ) -> str:
        """
        Produce a short, generic human-readable feedback string from
        the current metrics. Deliberately exercise-agnostic -- talks
        about movement quality in general terms only. Exercise-specific
        phrasing (e.g. "extend your fingers further") belongs in
        app/models/exercise_rules.py, layered on top of this output.

        Args:
            motion_score: Score from calculate_motion_score.
            tremor: Tremor index for the current window.
            paused: Whether a sustained pause was detected.
            in_motion: Whether the tracked point is currently moving.

        Returns:
            Short feedback sentence.
        """
        if paused:
            return "Movement paused. Resume when ready."
        if motion_score >= 80:
            return "Great control -- movement is smooth and stable."
        if motion_score >= 50:
            return "Movement detected, but there is some instability."
        if in_motion:
            return "Significant tremor detected -- try to slow down."
        return "Waiting for movement."

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def analyze_frame(
        self, raw_landmarks: Sequence[RawLandmark], timestamp: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Analyze a single frame of landmarks and update internal history.

        Args:
            raw_landmarks: Landmarks for this frame, either as
                (x, y[, z]) tuples or MediaPipe-style landmark objects.
            timestamp: Frame timestamp in seconds. Defaults to
                time.time() if not supplied (e.g. when the detector
                does not provide capture timestamps).

        Returns:
            Dictionary with keys: "angles", "velocity", "acceleration",
            "tremor", "repetitions", "motion_score", "range_of_motion",
            "in_motion", "paused", "feedback". On internal error, a
            safe zeroed-out dictionary plus an "error" key is returned
            instead of raising, so a single bad frame never crashes the
            video loop.
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

            tremor = self.detect_tremor()

            # "Primary" joint = first configured joint that produced an
            # angle this frame. Exercise-specific logic can later choose
            # a specific joint explicitly instead of relying on this.
            primary_joint = next(iter(angles), None)
            if primary_joint is not None:
                repetitions = self.count_repetitions(angles[primary_joint])
                rom = self.evaluate_range_of_motion(primary_joint)
            else:
                repetitions = self._rep_count
                rom = None

            motion_score = self.calculate_motion_score(tremor, rom)
            feedback = self.generate_analysis(motion_score, tremor, paused, in_motion)

            return {
                "angles": angles,
                "velocity": velocity,
                "acceleration": acceleration,
                "tremor": tremor,
                "repetitions": repetitions,
                "motion_score": motion_score,
                "range_of_motion": rom.range_of_motion if rom is not None else 0.0,
                "in_motion": in_motion,
                "paused": paused,
                "feedback": feedback,
            }
        except Exception as exc:  # noqa: BLE001 - deliberate top-level safety net
            logger.exception("analyze_frame failed: %s", exc)
            return {
                "angles": {},
                "velocity": 0.0,
                "acceleration": 0.0,
                "tremor": 0.0,
                "repetitions": self._rep_count,
                "motion_score": 0.0,
                "range_of_motion": 0.0,
                "in_motion": False,
                "paused": False,
                "feedback": "Analysis error.",
                "error": str(exc),
            }

    # ------------------------------------------------------------------
    # Session control
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """
        Clear all rolling history and counters. Call this at the start
        of a new exercise session/repetition set so metrics from a
        previous session don't leak into the new one.
        """
        self._landmark_history.clear()
        self._velocity_history.clear()
        self._joint_angle_histories.clear()
        self._rep_count = 0
        self._rep_direction = None
        self._last_extremum_angle = None
        self._pause_start_time = None
        logger.info("MotionAnalyzer state reset.")
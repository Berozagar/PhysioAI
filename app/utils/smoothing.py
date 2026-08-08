"""
===========================================
Landmark Smoothing Utility
-------------------------------------------
Uses Exponential Moving Average (EMA)
to reduce landmark jitter.
===========================================

Accepts landmarks in either form the project produces:

    - plain (x, y) / (x, y, z) tuples, as returned by
      PoseDetector.get_landmarks()
    - MediaPipe-style objects exposing .x / .y / .z

and always returns immutable SmoothedLandmark values. The input
landmarks are never modified in place, so a caller can keep using
its own raw landmarks after smoothing.
"""

from __future__ import annotations

from typing import Any, List, NamedTuple, Optional, Sequence

DEFAULT_ALPHA: float = 0.3


class SmoothedLandmark(NamedTuple):
    """
    One smoothed landmark.

    A NamedTuple so it satisfies every consumer in the project at
    once: attribute access (`lm.x`) for MediaPipe-style code, and
    tuple access/unpacking (`lm[0]`, `x, y, z = lm`) for the Core,
    which expects plain (x, y, z) tuples.
    """

    x: float
    y: float
    z: float


def _as_landmark(raw: Any) -> SmoothedLandmark:
    """
    Convert one raw landmark (tuple, list, or .x/.y/.z object) into a
    SmoothedLandmark.

    Raises:
        ValueError: If a tuple/list landmark has fewer than 2 coordinates.
        TypeError: If the landmark is neither a sequence nor an object
            exposing .x / .y.
    """
    if hasattr(raw, "x") and hasattr(raw, "y"):
        return SmoothedLandmark(
            float(raw.x), float(raw.y), float(getattr(raw, "z", 0.0))
        )

    if isinstance(raw, (tuple, list)):
        if len(raw) == 2:
            return SmoothedLandmark(float(raw[0]), float(raw[1]), 0.0)
        if len(raw) >= 3:
            return SmoothedLandmark(float(raw[0]), float(raw[1]), float(raw[2]))
        raise ValueError(f"Landmark tuple has invalid length: {raw!r}")

    raise TypeError(f"Unsupported landmark type: {type(raw)!r}")


class LandmarkSmoother:
    """
    Exponential moving average smoother for a full landmark list.

    Mathematics:
        smoothed = alpha * current + (1 - alpha) * previous_smoothed

    A lower alpha smooths harder but adds more lag; a higher alpha
    reacts faster but lets more jitter through.
    """

    def __init__(self, alpha: float = DEFAULT_ALPHA) -> None:
        """
        Args:
            alpha: EMA weight for the current frame, in (0.0, 1.0].
                1.0 disables smoothing entirely.

        Raises:
            ValueError: If alpha is outside (0.0, 1.0].
        """
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in the range (0.0, 1.0].")
        self.alpha = alpha
        self.previous: Optional[List[SmoothedLandmark]] = None

    def reset(self) -> None:
        """Forget the previous frame, e.g. when tracking is lost or a new set starts."""
        self.previous = None

    def smooth(
        self, landmarks: Optional[Sequence[Any]]
    ) -> Optional[List[SmoothedLandmark]]:
        """
        Smooth one frame of landmarks against the running average.

        Args:
            landmarks: This frame's landmarks, or None if tracking was lost.

        Returns:
            None if `landmarks` is None. An empty list if the frame had no
            landmarks. Otherwise a list of SmoothedLandmark, same length
            and order as the input.
        """
        if landmarks is None:
            return None

        current = [_as_landmark(lm) for lm in landmarks]
        if not current:
            return []

        # First frame, or the landmark count changed (person left and
        # re-entered): seed the average instead of blending mismatched points.
        if self.previous is None or len(self.previous) != len(current):
            self.previous = current
            return current

        alpha = self.alpha
        inverse = 1.0 - alpha
        smoothed = [
            SmoothedLandmark(
                alpha * curr.x + inverse * prev.x,
                alpha * curr.y + inverse * prev.y,
                alpha * curr.z + inverse * prev.z,
            )
            for prev, curr in zip(self.previous, current)
        ]

        self.previous = smoothed
        return smoothed

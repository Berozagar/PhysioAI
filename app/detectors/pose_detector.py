"""
app/detectors/pose_detector.py

This module is the computer-vision "input layer" for PhysioAI.

Its only job is to:
    1. Take a webcam frame (a normal OpenCV BGR image).
    2. Run it through the MediaPipe Tasks Pose Landmarker.
    3. Hand back the 33 body landmarks in a clean, simple format that
       any other module (Core, exercise rules, etc.) can consume
       without needing to know anything about MediaPipe itself.

Nothing in here touches kinematics.py, motion_analyzer.py, or
audio_feedback.py. This file only detects and exposes landmarks.

MIGRATION NOTE (mediapipe >= 0.10.x):
This module uses the modern MediaPipe Tasks API
(mediapipe.tasks.python / mediapipe.tasks.python.vision) instead of
the legacy `mp.solutions.pose` API, which no longer exists in current
MediaPipe releases. It also does NOT import `mediapipe.solutions` for
drawing -- the skeleton overlay is drawn manually with OpenCV, using
a small local list of landmark connections, so there is zero
dependency on the removed Solutions API anywhere in this file.

Two output formats are provided, for two different audiences:

    get_landmarks()         -> List[Tuple[float, float, float]]
        Core-compatible. This is the exact shape MotionAnalyzer /
        Kinematics expect: a plain list of 33 (x, y, z) tuples,
        normalized 0.0-1.0, in MediaPipe's landmark order (0 = NOSE,
        11/12 = shoulders, 13/14 = elbows, 15/16 = wrists,
        23/24 = hips, 25/26 = knees, 27/28 = ankles, etc.).

    get_landmark_details()  -> List[Dict[str, Any]]
        Everything else: id, name, visibility, and optional pixel
        coordinates. Use this for debugging, on-screen text, logging,
        or future modules -- never for feeding the Core directly.

Beginner note: "landmarks" just means points on the body (like
"left shoulder" or "right knee") that MediaPipe finds for us in
every frame, each with an (x, y, z) position and a confidence score.
"""

import os
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks_python
from mediapipe.tasks.python import vision as mp_tasks_vision

# ---------------------------------------------------------------------------
# Static reference data: landmark names and skeleton connections.
#
# These are plain Python data -- NOT part of mp.solutions -- so we can draw
# and label landmarks without importing the legacy Solutions API. The order
# and indices match MediaPipe's standard 33-point BlazePose model, which is
# unchanged between the old Solutions API and the new Tasks API.
# ---------------------------------------------------------------------------

POSE_LANDMARK_NAMES: List[str] = [
    "NOSE", "LEFT_EYE_INNER", "LEFT_EYE", "LEFT_EYE_OUTER",
    "RIGHT_EYE_INNER", "RIGHT_EYE", "RIGHT_EYE_OUTER",
    "LEFT_EAR", "RIGHT_EAR", "MOUTH_LEFT", "MOUTH_RIGHT",
    "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW",
    "LEFT_WRIST", "RIGHT_WRIST", "LEFT_PINKY", "RIGHT_PINKY",
    "LEFT_INDEX", "RIGHT_INDEX", "LEFT_THUMB", "RIGHT_THUMB",
    "LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE", "RIGHT_KNEE",
    "LEFT_ANKLE", "RIGHT_ANKLE", "LEFT_HEEL", "RIGHT_HEEL",
    "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX",
]

# (start_index, end_index) pairs describing which landmarks are connected
# by a "bone" for skeleton drawing. This mirrors MediaPipe's standard
# POSE_CONNECTIONS set.
POSE_CONNECTIONS: List[Tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),
    (24, 26), (26, 28), (28, 30), (30, 32), (28, 32),
]

# Project-relative default model location:
#   PhysioAI/
#   ├── app/detectors/pose_detector.py   <- this file
#   └── models/pose_landmarker.task      <- default model path
_THIS_FILE = os.path.abspath(__file__)
_APP_DETECTORS_DIR = os.path.dirname(_THIS_FILE)      # .../app/detectors
_APP_DIR = os.path.dirname(_APP_DETECTORS_DIR)        # .../app
_PROJECT_ROOT = os.path.dirname(_APP_DIR)             # .../PhysioAI
DEFAULT_MODEL_PATH = os.path.join(_PROJECT_ROOT, "models", "pose_landmarker.task")

_MODEL_DOWNLOAD_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)


class PoseDetector:
    """
    Wraps the MediaPipe Tasks Pose Landmarker so the rest of PhysioAI
    never has to import or configure MediaPipe directly.

    Typical usage:

        detector = PoseDetector()
        frame = detector.find_pose(frame)          # draws skeleton (optional)
        core_landmarks = detector.get_landmarks()  # List[(x, y, z)] for Core
        detector.close()

    Or, using it as a context manager so cleanup happens automatically:

        with PoseDetector() as detector:
            frame = detector.find_pose(frame)
            core_landmarks = detector.get_landmarks()
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        num_poses: int = 1,
        min_pose_detection_confidence: float = 0.5,
        min_pose_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        """
        Set up the MediaPipe Tasks Pose Landmarker.

        Args:
            model_path: Path to a .task Pose Landmarker model file. If
                None, defaults to <project_root>/models/pose_landmarker.task.
            num_poses: Maximum number of people to detect at once. 1 is
                the right choice for a single-user rehab session.
            min_pose_detection_confidence: How confident the model must
                be to say "yes, there is a person here" (0.0-1.0).
            min_pose_presence_confidence: How confident the model must
                be that a detected pose is still present in-frame.
            min_tracking_confidence: How confident the model must be to
                keep tracking a person it already found (0.0-1.0).

        Raises:
            FileNotFoundError: If no .task model file exists at
                model_path (or the default path). The error message
                explains exactly how to fix this.
        """
        self.model_path = model_path or DEFAULT_MODEL_PATH

        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(
                "\n\nPoseDetector could not find a MediaPipe Pose Landmarker "
                f"model file at:\n    {self.model_path}\n\n"
                "To fix this:\n"
                "  1. Create a 'models' folder in your PhysioAI project root "
                "(next to 'app/', 'assets/', etc.) if it doesn't exist yet.\n"
                "  2. Download the model file with this command:\n"
                f'     curl -L -o "{self.model_path}" {_MODEL_DOWNLOAD_URL}\n'
                "     (PowerShell users can use "
                f'Invoke-WebRequest -Uri "{_MODEL_DOWNLOAD_URL}" '
                f'-OutFile "{self.model_path}")\n'
                "  3. Re-run your script.\n"
            )

        try:
            base_options = mp_tasks_python.BaseOptions(model_asset_path=self.model_path)
            options = mp_tasks_vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=mp_tasks_vision.RunningMode.VIDEO,
                num_poses=num_poses,
                min_pose_detection_confidence=min_pose_detection_confidence,
                min_pose_presence_confidence=min_pose_presence_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
            self.landmarker = mp_tasks_vision.PoseLandmarker.create_from_options(options)
        except Exception as error:
            raise RuntimeError(
                "Failed to initialize the MediaPipe Pose Landmarker. Check "
                "that 'mediapipe' is installed correctly and that the model "
                f"file at '{self.model_path}' is a valid, uncorrupted "
                ".task file."
            ) from error

        # Stores the raw MediaPipe Tasks result from the most recent frame,
        # so get_landmarks() / get_landmark_details() / pose_detected() can
        # read from it without re-running detection.
        self.results: Optional[Any] = None

        # VIDEO running mode requires strictly increasing millisecond
        # timestamps across calls. We derive these from a monotonic clock
        # so they always increase, even if the wall clock is adjusted.
        self._start_time_ns = time.monotonic_ns()
        self._last_timestamp_ms = -1

    def _next_timestamp_ms(self) -> int:
        """Return a monotonically increasing millisecond timestamp."""
        elapsed_ms = (time.monotonic_ns() - self._start_time_ns) // 1_000_000
        if elapsed_ms <= self._last_timestamp_ms:
            elapsed_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = elapsed_ms
        return elapsed_ms

    def find_pose(self, frame: np.ndarray, draw: bool = True) -> np.ndarray:
        """
        Run pose detection on one frame and (optionally) draw the
        skeleton on top of it.

        Args:
            frame: A single webcam frame as a BGR image (this is the
                format OpenCV uses by default when you read from a
                camera).
            draw: If True, draws the detected skeleton/landmarks
                directly onto the returned frame. Set False if you
                only want the landmark data with a clean image.

        Returns:
            The same frame, with the skeleton drawn on it if
            draw=True and a person was detected. If no person is
            detected, the frame is returned unchanged.

        Raises:
            ValueError: If frame is None or not a valid image array.
        """
        if frame is None or not isinstance(frame, np.ndarray):
            raise ValueError("find_pose() received an invalid frame (None or not an image array).")

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        timestamp_ms = self._next_timestamp_ms()
        self.results = self.landmarker.detect_for_video(mp_image, timestamp_ms)

        if draw and self.results.pose_landmarks:
            self._draw_landmarks(frame, self.results.pose_landmarks[0])

        return frame

    def _draw_landmarks(self, frame: np.ndarray, landmarks) -> None:
        """
        Draw the skeleton (joints + connecting lines) onto frame using
        plain OpenCV calls -- no mp.solutions.drawing_utils involved.
        """
        height, width = frame.shape[:2]
        points = [(int(lm.x * width), int(lm.y * height)) for lm in landmarks]

        for start_idx, end_idx in POSE_CONNECTIONS:
            cv2.line(frame, points[start_idx], points[end_idx], (0, 200, 0), 2)

        for point in points:
            cv2.circle(frame, point, 4, (0, 0, 255), -1)

    def pose_detected(self) -> bool:
        """
        Check whether a person/pose was found in the most recently
        processed frame.

        Returns:
            True if find_pose() has been called and a pose was found,
            False otherwise (including if find_pose() hasn't run yet).
        """
        return bool(self.results is not None and self.results.pose_landmarks)

    def get_landmarks(self) -> List[Tuple[float, float, float]]:
        """
        Return the 33 pose landmarks in the exact format the existing
        Core (kinematics.py / motion_analyzer.py) expects: a plain
        list of (x, y, z) float tuples, normalized 0.0-1.0, in
        MediaPipe's fixed landmark order.

        Call find_pose() first on the same frame before calling this.

        This is the ONLY method that should feed the Core. It
        deliberately returns nothing but bare tuples -- no ids, names,
        or visibility.

        Returns:
            An empty list if no pose was detected. Otherwise, exactly
            33 tuples, one per MediaPipe Pose landmark, e.g.:

                [
                    (0.4823, 0.3011, -0.512),   # 0  = NOSE
                    ...
                    (0.55, 0.61, -0.02),         # 11 = LEFT_SHOULDER
                    ...
                ]
        """
        if not self.pose_detected():
            return []

        landmarks = self.results.pose_landmarks[0]
        return [(lm.x, lm.y, lm.z) for lm in landmarks]

    def get_landmark_details(
        self, frame: Optional[np.ndarray] = None
    ) -> List[Dict[str, Any]]:
        """
        Return the 33 pose landmarks with full detail: id, name,
        x/y/z, visibility, and (optionally) pixel coordinates.

        This is NOT Core-compatible on its own (the Core does not
        accept dictionaries) -- it's meant for debugging, on-screen
        overlays, logging, or future modules that want more than raw
        coordinates. Use get_landmarks() to feed the Core.

        Call find_pose() first on the same frame before calling this.

        Args:
            frame: Optional. Pass the same frame you gave to
                find_pose() if you also want pixel coordinates
                (x_px, y_px) included, since MediaPipe's raw
                coordinates are normalized (0.0-1.0) relative to
                image width/height, not actual pixels.

        Returns:
            An empty list if no pose was detected. Otherwise, a list
            of 33 dictionaries (one per landmark).
        """
        if not self.pose_detected():
            return []

        height: Optional[int] = None
        width: Optional[int] = None
        if frame is not None:
            height, width = frame.shape[:2]

        landmarks = self.results.pose_landmarks[0]
        details: List[Dict[str, Any]] = []
        for idx, lm in enumerate(landmarks):
            entry: Dict[str, Any] = {
                "id": idx,
                "name": POSE_LANDMARK_NAMES[idx],
                "x": lm.x,
                "y": lm.y,
                "z": lm.z,
                "visibility": getattr(lm, "visibility", None),
            }
            if width is not None and height is not None:
                entry["x_px"] = int(lm.x * width)
                entry["y_px"] = int(lm.y * height)
            details.append(entry)

        return details

    def close(self) -> None:
        """
        Release MediaPipe's internal resources. Always call this when
        you are done with the detector (or use the `with` statement,
        which calls it for you automatically).
        """
        self.landmarker.close()

    def __enter__(self) -> "PoseDetector":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
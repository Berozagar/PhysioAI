"""
run.py
======

PhysioAI application entrypoint.

Wires every module in the project into one real-time loop:

    VideoCapture ──> PoseDetector ──> LandmarkSmoother ──> MotionAnalyzer
                                                                 │
                                           ExerciseRule ◄────────┤
                                           PostureRules ◄────────┘
                                                 │
                                   Drawer overlay + AudioFeedback

Usage
-----
    python run.py                          # webcam 0, shoulder_raise
    python run.py --exercise squat
    python run.py --source clip.mp4        # analyse a recorded video
    python run.py --self-test              # full pipeline, no camera needed
    python run.py --list-exercises

Controls (live window)
----------------------
    q  quit          r  reset reps          m  mute / unmute audio
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from app.core.audio_feedback import AudioFeedback, FeedbackPriority
from app.core.motion_analyzer import MotionAnalyzer
from app.detectors.pose_detector import PoseDetector
from app.models.exercise_rules import EXERCISES, ExerciseRule
from app.models.posture_rules import PostureRules
from app.utils.drawing import Drawer
from app.utils.fps import FPSCounter
from app.utils.smoothing import LandmarkSmoother
from app.utils.synthetic import create_pose_landmarks, cycle_for_rule

logger = logging.getLogger("physioai")

WINDOW_NAME = "PhysioAI"
DEFAULT_EXERCISE = "shoulder_raise"

# PostureRules.evaluate() indexes these keys directly, so a frame is only
# posture-checked once every joint its exercise needs was computed.
REQUIRED_POSTURE_JOINTS: Dict[str, Tuple[str, ...]] = {
    "shoulder_raise": ("shoulder", "back", "neck"),
    "squat": ("knee", "back"),
    "knee_bend": ("knee",),
}

NO_POSE_FEEDBACK = "No pose detected - step into frame."
SELF_TEST_FRAME_INTERVAL = 0.05  # simulated 20 FPS


class PhysioSession:
    """
    Owns all analysis state for one exercise session.

    Deliberately knows nothing about cameras or windows: it takes a list
    of landmarks and returns everything needed to render and speak. That
    keeps the same code path exercised by live video and by --self-test.
    """

    def __init__(
        self,
        exercise: str,
        audio: Optional[AudioFeedback] = None,
        smoothing_alpha: Optional[float] = 0.3,
    ) -> None:
        """
        Args:
            exercise: Key from app.models.exercise_rules.EXERCISES.
            audio: Optional AudioFeedback instance. None disables audio.
            smoothing_alpha: EMA alpha for landmark smoothing, or None
                to disable smoothing entirely.
        """
        self.exercise = exercise
        self.rule = ExerciseRule(exercise)
        self.analyzer = MotionAnalyzer()
        self.posture = PostureRules()
        self.drawer = Drawer()
        self.audio = audio
        self.smoother = (
            LandmarkSmoother(alpha=smoothing_alpha) if smoothing_alpha else None
        )

        self._required_joints = REQUIRED_POSTURE_JOINTS.get(exercise, ())
        self._last_reps = 0
        self._last_spoken: Optional[str] = None
        self._muted = False

        if not self._required_joints:
            logger.warning(
                "No posture joints registered for '%s'; posture checks are "
                "disabled. Add it to REQUIRED_POSTURE_JOINTS in run.py and to "
                "PostureRules.evaluate() to enable them.",
                exercise,
            )

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------
    def analyze(
        self, landmarks: Sequence[Any], timestamp: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Run one frame of landmarks through the full analysis stack.

        Args:
            landmarks: 33 landmarks as (x, y, z) tuples or MediaPipe objects.
            timestamp: Frame time in seconds; defaults to time.time().

        Returns:
            The MotionAnalyzer result dict, plus "posture_ok" and
            "posture_feedback" keys.
        """
        if self.smoother is not None:
            landmarks = self.smoother.smooth(landmarks) or []

        result = self.analyzer.analyze_frame(
            landmarks, timestamp=timestamp, exercise_rule=self.rule
        )

        posture_ok, posture_feedback = self._check_posture(result.get("angles", {}))
        result["posture_ok"] = posture_ok
        result["posture_feedback"] = posture_feedback

        self._handle_audio(result)
        return result

    def _check_posture(self, angles: Dict[str, float]) -> Tuple[bool, List[str]]:
        """
        Evaluate PostureRules when every joint it needs is available.

        Returns:
            (posture_ok, messages). Falls back to (True, []) when the
            required joints were not all computed this frame.
        """
        if not self._required_joints:
            return True, []
        if any(joint not in angles for joint in self._required_joints):
            return True, []

        verdict = self.posture.evaluate(self.exercise, angles)
        return bool(verdict["correct"]), list(verdict["feedback"])

    def idle_result(self) -> Dict[str, Any]:
        """Result placeholder for frames where no pose was detected."""
        if self.smoother is not None:
            self.smoother.reset()
        return {
            "angles": {},
            "repetitions": self.rule.get_count(),
            "stage": self.rule.get_stage(),
            "progress": 0.0,
            "feedback": NO_POSE_FEEDBACK,
            "posture_ok": True,
            "posture_feedback": [],
        }

    # ------------------------------------------------------------------
    # Audio
    # ------------------------------------------------------------------
    def _handle_audio(self, result: Dict[str, Any]) -> None:
        """Play a chime on each completed rep and speak feedback as it changes."""
        if self.audio is None:
            return

        reps = int(result.get("repetitions") or 0)
        if reps > self._last_reps:
            self.audio.play_success()
            self.audio.speak(f"{reps}", FeedbackPriority.HIGH)
        self._last_reps = reps

        feedback = result.get("feedback")
        if feedback and feedback != self._last_spoken:
            self.audio.speak(feedback)
            self._last_spoken = feedback

    # ------------------------------------------------------------------
    # Rendering / lifecycle
    # ------------------------------------------------------------------
    def render(self, frame: np.ndarray, result: Dict[str, Any], fps: float) -> np.ndarray:
        """Draw the information panel for `result` onto `frame`."""
        return self.drawer.draw_information(
            frame=frame,
            exercise=self.exercise,
            reps=result.get("repetitions", 0),
            stage=result.get("stage"),
            progress=result.get("progress"),
            feedback=result.get("feedback", ""),
            fps=fps,
            posture_feedback=result.get("posture_feedback"),
            posture_ok=result.get("posture_ok"),
        )

    def toggle_mute(self) -> bool:
        """
        Toggle all audio off and back on.

        Returns:
            True if audio is now muted, False if it is audible. Always
            False when the session has no audio engine.
        """
        if self.audio is None:
            return False

        self._muted = not self._muted
        self.audio.enable_voice(not self._muted)
        self.audio.enable_sound_effects(not self._muted)
        if self._muted:
            self.audio.stop()  # drop anything already queued
        logger.info("Audio %s.", "muted" if self._muted else "unmuted")
        return self._muted

    def reset(self) -> None:
        """Clear reps and all rolling history, e.g. to start a fresh set."""
        self.rule.reset()
        self.analyzer.reset()
        if self.smoother is not None:
            self.smoother.reset()
        self._last_reps = 0
        self._last_spoken = None
        logger.info("Session reset.")


# ----------------------------------------------------------------------
# Video helpers
# ----------------------------------------------------------------------
def candidate_backends() -> List[Tuple[int, str]]:
    """
    Capture backends to try, most reliable first for this platform.

    On Windows the default (Media Foundation) is often slow to open or
    fails outright on webcams that DirectShow handles fine, so DirectShow
    is tried first and the others are kept as fallbacks.
    """
    if sys.platform == "win32":
        return [
            (cv2.CAP_DSHOW, "DirectShow"),
            (cv2.CAP_MSMF, "Media Foundation"),
            (cv2.CAP_ANY, "default"),
        ]
    return [(cv2.CAP_ANY, "default")]


def open_camera(
    index: int, width: int, height: int
) -> Tuple[Optional[cv2.VideoCapture], Optional[str]]:
    """
    Open a webcam, trying each backend until one both opens and delivers
    a frame.

    isOpened() returning True is not sufficient -- some drivers report a
    camera as open and then fail on every read -- so each candidate is
    confirmed by actually decoding one frame.

    Returns:
        (capture, backend_name), or (None, None) if no backend worked.
    """
    for backend, name in candidate_backends():
        cap = cv2.VideoCapture(index, backend)
        if not cap.isOpened():
            cap.release()
            continue

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        # Keep only the newest frame so analysis tracks the live pose
        # instead of drifting behind a backlog of buffered frames.
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if cap.read()[0]:
            actual = (
                int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            )
            if actual != (width, height):
                logger.info(
                    "Camera %d supplied %dx%d instead of the requested %dx%d.",
                    index, actual[0], actual[1], width, height,
                )
            logger.info("Camera %d opened via %s.", index, name)
            return cap, name

        cap.release()

    return None, None


def list_cameras(width: int, height: int, max_index: int = 5) -> List[Tuple[int, str]]:
    """
    Probe camera indices 0..max_index and report which ones deliver frames.

    Returns:
        List of (index, backend_name) for every working camera.
    """
    working: List[Tuple[int, str]] = []
    for index in range(max_index + 1):
        cap, backend = open_camera(index, width, height)
        if cap is not None:
            working.append((index, backend or "unknown"))
            cap.release()
    return working


def open_capture(source: str, width: int, height: int) -> Optional[cv2.VideoCapture]:
    """
    Open a webcam index (e.g. "0") or a video file path.

    Returns:
        An opened VideoCapture, or None if the source could not be opened.
    """
    if source.isdigit():
        return open_camera(int(source), width, height)[0]

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        cap.release()
        return None
    return cap


def build_audio(args: argparse.Namespace) -> Optional[AudioFeedback]:
    """Create an AudioFeedback instance unless audio was disabled."""
    if args.no_audio:
        return None
    return AudioFeedback(
        voice_enabled=not args.no_voice,
        sound_effects_enabled=not args.no_sfx,
    )


def handle_key(key: int, session: PhysioSession) -> bool:
    """
    Apply a keypress. Returns False when the user asked to quit.
    """
    if key in (ord("q"), 27):  # q or Esc
        return False
    if key == ord("r"):
        session.reset()
    elif key == ord("m"):
        print("Audio muted." if session.toggle_mute() else "Audio unmuted.")
    return True


# ----------------------------------------------------------------------
# Modes
# ----------------------------------------------------------------------
def run_live(args: argparse.Namespace) -> int:
    """Run the pipeline against a webcam or video file. Returns an exit code."""
    cap = open_capture(args.source, args.width, args.height)
    if cap is None:
        print(
            f"ERROR: could not open video source {args.source!r}.\n"
            "  - For a webcam, run 'python run.py --list-cameras' to see which\n"
            "    indices work, and close any app already using the camera.\n"
            "  - For a file, check the path exists.\n"
            "  - With no camera available, try: python run.py --self-test",
            file=sys.stderr,
        )
        return 1

    try:
        detector = PoseDetector(model_path=args.model)
    except FileNotFoundError as error:
        print(str(error), file=sys.stderr)
        cap.release()
        return 1

    session = PhysioSession(args.exercise, build_audio(args), args.smoothing)
    fps_counter = FPSCounter()
    mirror = args.source.isdigit() and not args.no_mirror
    frames = 0
    last_frame: Optional[np.ndarray] = None

    print(f"Running '{args.exercise}' on source {args.source!r}. Press 'q' to quit.")

    try:
        with detector:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if mirror:
                    frame = cv2.flip(frame, 1)

                frame = detector.find_pose(frame, draw=True)
                fps = fps_counter.update()

                if detector.pose_detected():
                    result = session.analyze(detector.get_landmarks())
                else:
                    result = session.idle_result()

                last_frame = session.render(frame, result, fps)
                frames += 1

                if not args.headless:
                    cv2.imshow(WINDOW_NAME, last_frame)
                    if not handle_key(cv2.waitKey(1) & 0xFF, session):
                        break
                if args.max_frames and frames >= args.max_frames:
                    break
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        cap.release()
        if not args.headless:
            cv2.destroyAllWindows()
        if session.audio is not None:
            session.audio.shutdown()

    _report(session, frames, last_frame, args.save_frame)
    return 0


def run_self_test(args: argparse.Namespace) -> int:
    """
    Drive the full pipeline with synthetic landmarks -- no camera, no
    model, no window. Verifies wiring and rep counting end to end.
    """
    session = PhysioSession(args.exercise, build_audio(args), args.smoothing)
    fps_counter = FPSCounter()

    # Derive the sweep from the selected rule's own thresholds, so any
    # exercise added to EXERCISES self-tests correctly with no changes here.
    rule = EXERCISES[args.exercise]
    angles = cycle_for_rule(
        rule["start_angle"], rule["end_angle"], args.self_test_reps
    )
    is_knee = rule["joint"] == "knee"

    tracked_joint = EXERCISES[args.exercise]["joint"]
    print(f"Self-test: '{args.exercise}', {len(angles)} synthetic frames, "
          f"expecting {args.self_test_reps} reps.")
    print("Reporting the MEASURED joint angle (after smoothing), not the input angle.\n")

    last_frame: Optional[np.ndarray] = None
    timestamp = time.time()
    previous_state: Optional[Tuple[int, Optional[str]]] = None

    for index, angle in enumerate(angles):
        landmarks = create_pose_landmarks(
            shoulder_angle_deg=180.0 if is_knee else angle,
            knee_angle_deg=angle if is_knee else 180.0,
            nose_y=0.20 + 0.002 * (index % 5),
        )
        timestamp += SELF_TEST_FRAME_INTERVAL
        result = session.analyze(landmarks, timestamp=timestamp)

        frame = np.zeros((args.height, args.width, 3), dtype=np.uint8)
        last_frame = session.render(frame, result, fps_counter.update())

        # Print on every state change plus a periodic heartbeat, so a long
        # sweep stays readable but no repetition or stage flip is hidden.
        state = (int(result["repetitions"]), result["stage"])
        if state != previous_state or index % 10 == 0 or index == len(angles) - 1:
            measured = result["angles"].get(tracked_joint)
            measured_text = "  n/a" if measured is None else f"{measured:6.1f}"
            print(f"  frame {index + 1:>3}  input={angle:6.1f}  measured={measured_text}  "
                  f"reps={state[0]}  stage={str(state[1]):<8} "
                  f"posture_ok={str(result['posture_ok']):<5} {result['feedback']}")
        previous_state = state

    if session.audio is not None:
        session.audio.shutdown()

    reps = session.rule.get_count()
    _report(session, len(angles), last_frame, args.save_frame)

    if reps != args.self_test_reps:
        print(f"\nSELF-TEST FAILED: expected {args.self_test_reps} reps, counted {reps}.")
        return 1

    print("\nSELF-TEST PASSED: pipeline wired correctly and rep count is exact.")
    return 0


def _report(
    session: PhysioSession,
    frames: int,
    last_frame: Optional[np.ndarray],
    save_frame: Optional[str],
) -> None:
    """Print the end-of-session summary and optionally save the last overlay."""
    print(f"\nSession summary: exercise={session.exercise} "
          f"reps={session.rule.get_count()} frames={frames}")

    if save_frame and last_frame is not None:
        # imwrite reports failure by return value rather than raising -- most
        # commonly because the target folder does not exist. Without this
        # check the run claims to have saved a file that was never written.
        if cv2.imwrite(save_frame, last_frame):
            print(f"Saved last annotated frame to {save_frame}")
        else:
            print(
                f"WARNING: could not write {save_frame!r} "
                "(check the folder exists and the extension is supported).",
                file=sys.stderr,
            )


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="PhysioAI - real-time physiotherapy exercise coach.",
    )
    parser.add_argument("--exercise", choices=sorted(EXERCISES), default=DEFAULT_EXERCISE)
    parser.add_argument("--source", default="0",
                        help="Webcam index (e.g. 0) or path to a video file.")
    parser.add_argument("--model", default=None, help="Override the .task model path.")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--smoothing", type=float, default=0.3,
                        help="EMA alpha in (0,1]. Use 0 to disable smoothing.")
    parser.add_argument("--no-mirror", action="store_true",
                        help="Do not horizontally flip webcam frames.")
    parser.add_argument("--no-audio", action="store_true", help="Disable all audio.")
    parser.add_argument("--no-voice", action="store_true", help="Disable spoken feedback.")
    parser.add_argument("--no-sfx", action="store_true", help="Disable sound effects.")
    parser.add_argument("--headless", action="store_true",
                        help="Do not open a window (no display / CI).")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="Stop after N frames. 0 means unlimited.")
    parser.add_argument("--save-frame", default=None,
                        help="Write the last annotated frame to this image path.")
    parser.add_argument("--self-test", action="store_true",
                        help="Run the pipeline on synthetic data, no camera needed.")
    parser.add_argument("--self-test-reps", type=int, default=2)
    parser.add_argument("--list-exercises", action="store_true")
    parser.add_argument("--list-cameras", action="store_true",
                        help="Probe camera indices 0-5 and report which ones work.")
    parser.add_argument("--log-level", default="WARNING",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point. Returns a process exit code."""
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s %(name)s: %(message)s",
    )

    if args.list_cameras:
        cameras = list_cameras(args.width, args.height)
        if not cameras:
            print("No working cameras found on indices 0-5.\n"
                  "Plug one in, close any app using it, then try again.\n"
                  "Meanwhile: python run.py --self-test")
            return 1
        for index, backend in cameras:
            print(f"camera {index}  backend={backend}   use: python run.py --source {index}")
        return 0

    if args.list_exercises:
        for name, rule in sorted(EXERCISES.items()):
            print(f"{name:<16} joint={rule['joint']:<9} "
                  f"{rule['start_angle']} -> {rule['end_angle']} degrees")
        return 0

    # argparse gives 0.0 for "disabled"; LandmarkSmoother requires (0, 1].
    args.smoothing = args.smoothing if args.smoothing and args.smoothing > 0 else None

    if args.self_test:
        return run_self_test(args)
    return run_live(args)


if __name__ == "__main__":
    sys.exit(main())

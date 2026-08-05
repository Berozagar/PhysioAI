"""
PhysioAI Core Integration Test
================================

This file tests the three Core modules together, plus their
integration with Avanti's exercise-rules module:

1. Kinematics
2. Motion Analyzer
3. Audio Feedback
4. app.models.exercise_rules.ExerciseRule (integration point)

The synthetic data below simulates a real supported exercise
(shoulder_raise, see app/models/exercise_rules.py) using 33-point
MediaPipe Pose-shaped landmarks -- the same shape PoseDetector.get_
landmarks() produces -- rather than hand landmarks, since PhysioAI's
only detector is a body-pose detector, not a hand detector.

Run from the PhysioAI root folder:

    python test_core.py
"""

import math
import time

from app.core.kinematics import (
    calculate_angle,
    calculate_velocity,
    calculate_acceleration,
    calculate_range_of_motion,
    shoulder_angle,
    knee_angle,
)

from app.core.motion_analyzer import MotionAnalyzer

from app.core.audio_feedback import AudioFeedback

from app.models.exercise_rules import ExerciseRule


# ============================================================
# Helper function
# ============================================================

# MediaPipe Pose landmark indices used below (left side of the body).
NUM_POSE_LANDMARKS = 33
LEFT_EAR = 7
LEFT_SHOULDER = 11
LEFT_ELBOW = 13
LEFT_WRIST = 15
LEFT_HIP = 23
LEFT_KNEE = 25
LEFT_ANKLE = 27


def create_test_pose_landmarks(shoulder_angle_deg, nose_y=0.20):
    """
    Create 33 synthetic MediaPipe Pose-style body landmarks
    representing one moment of a left-arm shoulder-raise exercise.

    The landmark set MotionAnalyzer's default joint_config actually
    reads is:
        shoulder -> (hip=23, shoulder=11, elbow=13)
        elbow    -> (shoulder=11, elbow=13, wrist=15)
        knee     -> (hip=23, knee=25, ankle=27)
        back     -> (shoulder=11, hip=23, knee=25)
        neck     -> (ear=7, shoulder=11, hip=23)

    Hip, elbow-extension, knee, ear, and ankle positions are fixed;
    only the shoulder angle (elbow position) varies between frames,
    since that's the joint app/models/exercise_rules.py's
    "shoulder_raise" configuration tracks.

    Args:
        shoulder_angle_deg: Desired angle (degrees) at the shoulder
            between the hip and the elbow, i.e. the value
            kinematics.shoulder_angle() would return for this frame.
        nose_y: Slight, frame-varying nose y-position so the reference
            landmark (index 0) shows a small amount of overall body
            motion, exercising MotionAnalyzer's velocity/pause logic.

    Returns:
        List of 33 (x, y, z) tuples in MediaPipe Pose landmark order.
    """
    landmarks = [(0.5 + i * 0.001, 0.5 + i * 0.001, 0.0) for i in range(NUM_POSE_LANDMARKS)]

    hip = (0.50, 0.90, 0.0)
    theta = math.radians(shoulder_angle_deg)
    # elbow is placed so that calculate_angle(hip, shoulder, elbow) at
    # the shoulder vertex equals shoulder_angle_deg.
    shoulder = (0.50, 0.60, 0.0)
    elbow = (
        shoulder[0] + 0.30 * math.sin(theta),
        shoulder[1] + 0.30 * math.cos(theta),
        0.0,
    )
    wrist = (elbow[0], elbow[1] - 0.10, 0.0)
    knee = (0.50, 1.20, 0.0)
    ankle = (0.50, 1.50, 0.0)
    ear = (0.48, 0.50, 0.0)
    nose = (0.50, nose_y, 0.0)

    landmarks[0] = nose
    landmarks[LEFT_EAR] = ear
    landmarks[LEFT_SHOULDER] = shoulder
    landmarks[LEFT_ELBOW] = elbow
    landmarks[LEFT_WRIST] = wrist
    landmarks[LEFT_HIP] = hip
    landmarks[LEFT_KNEE] = knee
    landmarks[LEFT_ANKLE] = ankle

    return landmarks


# ============================================================
# MAIN TEST
# ============================================================

def main():

    print("=" * 60)
    print("        PHYSIOAI CORE INTEGRATION TEST")
    print("=" * 60)

    # ========================================================
    # TEST 1: KINEMATICS
    # ========================================================

    print("\n[1] Testing Kinematics...")

    try:

        # Generic angle primitive
        angle = calculate_angle((0, 1, 0), (0, 0, 0), (1, 0, 0))
        print(f"    Angle: {angle:.2f} degrees")

        # Pose-specific angle wrappers
        shoulder_test_angle = shoulder_angle((0.5, 0.9, 0.0), (0.5, 0.6, 0.0), (0.8, 0.6, 0.0))
        print(f"    Shoulder angle: {shoulder_test_angle:.2f} degrees")

        knee_test_angle = knee_angle((0.5, 0.9, 0.0), (0.5, 1.2, 0.0), (0.5, 1.5, 0.0))
        print(f"    Knee angle: {knee_test_angle:.2f} degrees")

        # Velocity
        velocity = calculate_velocity((0, 0, 0), (1, 0, 0), 1.0)
        print(f"    Velocity: {velocity:.2f}")

        # Acceleration
        acceleration = calculate_acceleration(1.0, 3.0, 1.0)
        print(f"    Acceleration: {acceleration:.2f}")

        # ROM
        rom = calculate_range_of_motion([30, 45, 60, 90])
        print(f"    Minimum Angle: {rom.min_angle:.2f}")
        print(f"    Maximum Angle: {rom.max_angle:.2f}")
        print(f"    Range of Motion: {rom.range_of_motion:.2f}")

        print("    Kinematics: PASS")

    except Exception as e:

        print("    Kinematics: FAIL")
        print(f"    Error: {e}")

    # ========================================================
    # TEST 2: MOTION ANALYZER + EXERCISE RULES INTEGRATION
    # ========================================================

    print("\n[2] Testing Motion Analyzer (shoulder_raise exercise)...")

    analyzer = MotionAnalyzer()
    exercise = ExerciseRule("shoulder_raise")

    results = []

    try:

        # Simulate one full shoulder-raise repetition: arm starts down
        # (~20 degrees), rises to fully raised (~165 degrees), then
        # returns back down (~20 degrees). app/models/exercise_rules.py's
        # "shoulder_raise" config uses start_angle=30, end_angle=160, so
        # this trace should register exactly one repetition.
        shoulder_angles_deg = [20, 60, 100, 140, 165, 150, 100, 60, 20]

        start_time = time.time()
        frame_interval = 1.0 / 30.0  # simulate ~30 FPS

        for i, angle_deg in enumerate(shoulder_angles_deg):

            landmarks = create_test_pose_landmarks(
                shoulder_angle_deg=angle_deg,
                nose_y=0.20 - (angle_deg * 0.0002),
            )
            timestamp = start_time + (i * frame_interval)

            result = analyzer.analyze_frame(
                landmarks,
                timestamp=timestamp,
                exercise_rule=exercise,
            )
            results.append(result)

            print(
                f"    Frame {i + 1}: "
                f"ShoulderAngle={result['angles'].get('shoulder', float('nan')):.1f}, "
                f"Reps={result['repetitions']}, "
                f"Stage={result['stage']}, "
                f"Quality={result['movement_quality']:.1f}, "
                f"Feedback='{result['feedback']}'"
            )

        final_result = results[-1]

        print("\n    Final Motion Analysis:")
        print(f"    Angles: {final_result['angles']}")
        print(f"    Velocity: {final_result['velocity']:.4f}")
        print(f"    Acceleration: {final_result['acceleration']:.4f}")
        print(f"    Smoothness: {final_result['smoothness']:.2f}")
        print(f"    Repetitions: {final_result['repetitions']}")
        print(f"    Range of Motion: {final_result['range_of_motion']:.4f}")
        print(f"    Movement Quality: {final_result['movement_quality']:.2f}")
        print(f"    In Motion: {final_result['in_motion']}")
        print(f"    Paused: {final_result['paused']}")
        print(f"    Feedback: {final_result['feedback']}")

        assert final_result["repetitions"] >= 1, (
            "Expected at least one repetition to be counted for a full "
            "raise-and-lower cycle."
        )
        assert final_result["range_of_motion"] > 0, (
            "Expected range of motion to reflect the actual angle swing, "
            "not remain at 0."
        )

        forbidden_terms = ["tremor", "neurohand", "parkinson", "neurological"]
        all_feedback_text = " ".join(r["feedback"].lower() for r in results)
        for term in forbidden_terms:
            assert term not in all_feedback_text, (
                f"Feedback unexpectedly contains forbidden term: '{term}'"
            )

        if "error" in final_result:
            print("\n    Motion Analyzer returned an error:")
            print(f"    {final_result['error']}")
        else:
            print("\n    Motion Analyzer: PASS")

    except Exception as e:

        print("    Motion Analyzer: FAIL")
        print(f"    Error: {e}")

    # ========================================================
    # TEST 3: AUDIO FEEDBACK
    # ========================================================

    print("\n[3] Testing Audio Feedback...")

    audio = None

    try:

        audio = AudioFeedback(
            voice_enabled=True,
            sound_effects_enabled=False,
        )
        print("    AudioFeedback object created successfully.")

        feedback_message = results[-1]["feedback"] if results else "Repetition completed."
        print(f"    Sending feedback: '{feedback_message}'")

        assert "tremor" not in feedback_message.lower(), (
            "Feedback sent to AudioFeedback must not reference tremor."
        )

        audio.speak(feedback_message)

        # Give the background speech thread a moment to process the message.
        time.sleep(2)

        print("    Audio Feedback: PASS")

    except Exception as e:

        print("    Audio Feedback: FAIL")
        print(f"    Error: {e}")

    finally:

        if audio is not None:
            audio.shutdown()

    # ========================================================
    # FINAL RESULT
    # ========================================================

    print("\n" + "=" * 60)
    print("        CORE INTEGRATION TEST COMPLETE")
    print("=" * 60)

    print("\nYour Core pipeline is:")

    print(
        """
    Body-Pose Landmarks (PoseDetector-shaped)
            |
            v
      Motion Analyzer  <---- Kinematics (angles, velocity, ROM, smoothness)
            |
            v
   ExerciseRule (app/models/exercise_rules.py)
            |
            v
    Repetitions, Stage, Progress, Feedback
            |
            v
       Audio Feedback
            |
            v
       Voice Output
    """
    )

    print("=" * 60)


# ============================================================
# RUN PROGRAM
# ============================================================

if __name__ == "__main__":
    main()
"""
NeuroHand Core Integration Test
================================

This file tests the three Core modules together:

1. Kinematics
2. Motion Analyzer
3. Audio Feedback

Run from the NeuroHand root folder:

    python test_core.py
"""

import time

from app.core.kinematics import (
    calculate_angle,
    calculate_velocity,
    calculate_acceleration,
    calculate_range_of_motion,
)

from app.core.motion_analyzer import MotionAnalyzer

from app.core.audio_feedback import AudioFeedback


# ============================================================
# Helper function
# ============================================================

def create_test_landmarks(wrist_x, wrist_y):
    """
    Create 21 fake MediaPipe-style hand landmarks.

    The important landmarks for the current MotionAnalyzer are:
        0  -> Wrist
        5, 6, 8 -> Index finger
        9, 10, 12 -> Middle finger

    The remaining landmarks are filled with simple positions.
    """

    landmarks = []

    # Create 21 basic landmarks
    for i in range(21):
        landmarks.append(
            (
                wrist_x + (i * 0.01),
                wrist_y + (i * 0.005),
                0.0
            )
        )

    # --------------------------------------------------------
    # Wrist (landmark 0)
    # --------------------------------------------------------
    landmarks[0] = (
        wrist_x,
        wrist_y,
        0.0
    )

    # --------------------------------------------------------
    # Index finger landmarks
    # MotionAnalyzer uses (5, 6, 8)
    # --------------------------------------------------------

    landmarks[5] = (
        wrist_x + 0.05,
        wrist_y,
        0.0
    )

    landmarks[6] = (
        wrist_x + 0.10,
        wrist_y + 0.02,
        0.0
    )

    landmarks[8] = (
        wrist_x + 0.15,
        wrist_y,
        0.0
    )

    # --------------------------------------------------------
    # Middle finger landmarks
    # MotionAnalyzer uses (9, 10, 12)
    # --------------------------------------------------------

    landmarks[9] = (
        wrist_x + 0.03,
        wrist_y + 0.05,
        0.0
    )

    landmarks[10] = (
        wrist_x + 0.06,
        wrist_y + 0.10,
        0.0
    )

    landmarks[12] = (
        wrist_x + 0.09,
        wrist_y + 0.15,
        0.0
    )

    return landmarks


# ============================================================
# MAIN TEST
# ============================================================

def main():

    print("=" * 60)
    print("        NEUROHAND CORE INTEGRATION TEST")
    print("=" * 60)

    # ========================================================
    # TEST 1: KINEMATICS
    # ========================================================

    print("\n[1] Testing Kinematics...")

    try:

        # Test angle
        angle = calculate_angle(
            (0, 1, 0),
            (0, 0, 0),
            (1, 0, 0)
        )

        print(f"    Angle: {angle:.2f} degrees")

        # Test velocity
        velocity = calculate_velocity(
            (0, 0, 0),
            (1, 0, 0),
            1.0
        )

        print(f"    Velocity: {velocity:.2f}")

        # Test acceleration
        acceleration = calculate_acceleration(
            1.0,
            3.0,
            1.0
        )

        print(f"    Acceleration: {acceleration:.2f}")

        # Test ROM
        rom = calculate_range_of_motion(
            [30, 45, 60, 90]
        )

        print(f"    Minimum Angle: {rom.min_angle:.2f}")
        print(f"    Maximum Angle: {rom.max_angle:.2f}")
        print(f"    Range of Motion: {rom.range_of_motion:.2f}")

        print("    Kinematics: PASS")

    except Exception as e:

        print("    Kinematics: FAIL")
        print(f"    Error: {e}")

    # ========================================================
    # TEST 2: MOTION ANALYZER
    # ========================================================

    print("\n[2] Testing Motion Analyzer...")

    analyzer = MotionAnalyzer()

    try:

        # Simulate multiple frames.
        # Each frame represents a slightly different
        # position of the hand.

        frames = [
            create_test_landmarks(0.10, 0.50),
            create_test_landmarks(0.12, 0.50),
            create_test_landmarks(0.15, 0.50),
            create_test_landmarks(0.18, 0.50),
            create_test_landmarks(0.20, 0.50),
            create_test_landmarks(0.18, 0.50),
            create_test_landmarks(0.15, 0.50),
            create_test_landmarks(0.12, 0.50),
        ]

        results = []

        # Start timestamp
        start_time = time.time()

        # Simulate approximately 30 FPS
        frame_interval = 1.0 / 30.0

        for i, landmarks in enumerate(frames):

            timestamp = start_time + (i * frame_interval)

            result = analyzer.analyze_frame(
                landmarks,
                timestamp=timestamp
            )

            results.append(result)

            print(
                f"    Frame {i + 1}: "
                f"Velocity={result['velocity']:.4f}, "
                f"Score={result['motion_score']:.2f}"
            )

        # Get final result
        final_result = results[-1]

        print("\n    Final Motion Analysis:")
        print(
            f"    Angles: "
            f"{final_result['angles']}"
        )

        print(
            f"    Velocity: "
            f"{final_result['velocity']:.4f}"
        )

        print(
            f"    Acceleration: "
            f"{final_result['acceleration']:.4f}"
        )

        print(
            f"    Tremor: "
            f"{final_result['tremor']:.4f}"
        )

        print(
            f"    Repetitions: "
            f"{final_result['repetitions']}"
        )

        print(
            f"    Range of Motion: "
            f"{final_result['range_of_motion']:.4f}"
        )

        print(
            f"    Motion Score: "
            f"{final_result['motion_score']:.2f}"
        )

        print(
            f"    In Motion: "
            f"{final_result['in_motion']}"
        )

        print(
            f"    Paused: "
            f"{final_result['paused']}"
        )

        print(
            f"    Feedback: "
            f"{final_result['feedback']}"
        )

        if "error" in final_result:

            print(
                "\n    Motion Analyzer returned an error:"
            )

            print(
                f"    {final_result['error']}"
            )

        else:

            print(
                "\n    Motion Analyzer: PASS"
            )

    except Exception as e:

        print("    Motion Analyzer: FAIL")
        print(f"    Error: {e}")

    # ========================================================
    # TEST 3: AUDIO FEEDBACK
    # ========================================================

    print("\n[3] Testing Audio Feedback...")

    audio = None

    try:

        # Create AudioFeedback object
        audio = AudioFeedback(
            voice_enabled=True,
            sound_effects_enabled=False
        )

        print(
            "    AudioFeedback object created successfully."
        )

        # Use the feedback generated by MotionAnalyzer
        feedback_message = results[-1]["feedback"]

        print(
            f"    Sending feedback: "
            f"'{feedback_message}'"
        )

        # Speak the feedback
        audio.speak(feedback_message)

        # Give the background speech thread
        # a moment to process the message
        time.sleep(2)

        print(
            "    Audio Feedback: PASS"
        )

    except Exception as e:

        print("    Audio Feedback: FAIL")
        print(f"    Error: {e}")

    finally:

        # Always shut down the audio thread
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
    Fake Hand Landmarks
            |
            v
    Motion Analyzer
            |
            v
       Kinematics
            |
            v
    Motion Analysis Result
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
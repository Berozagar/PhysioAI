"""
test_pose_detector.py

Standalone test for app/detectors/pose_detector.py.

Purpose
-------
Confirms, independently of the Core, that this pipeline works:

    Webcam -> OpenCV -> MediaPipe Tasks Pose Landmarker -> 33 landmarks
        -> Core-compatible (x, y, z) tuples

This test does NOT import or call anything from app/core/*. It only
proves the detector itself is producing correct, Core-shaped output.

Controls
--------
Press 'q' in the video window to quit.
"""

import time

import cv2

from app.detectors.pose_detector import PoseDetector

WINDOW_NAME = "PhysioAI - Pose Detector Test"
PRINT_INTERVAL_SECONDS = 2.0  # how often to print sample data to the console


def main() -> None:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam (index 0). "
              "Check that no other app is using the camera.")
        return

    last_print_time = 0.0

    try:
        detector = PoseDetector()
    except FileNotFoundError as error:
        print(str(error))
        cap.release()
        return

    with detector:
        print("Webcam opened. Press 'q' in the video window to quit.\n")

        while True:
            success, frame = cap.read()
            if not success:
                print("WARNING: Could not read a frame from the webcam. Stopping.")
                break

            # Run detection and draw the skeleton on the frame.
            frame = detector.find_pose(frame, draw=True)

            if detector.pose_detected():
                # (1) Core-compatible output: exactly what MotionAnalyzer /
                #     Kinematics will receive once the Core is wired in.
                core_landmarks = detector.get_landmarks()

                # (2) Detailed output: for debugging/on-screen info only.
                details = detector.get_landmark_details(frame)

                # Sanity check: we should always get exactly 33 landmarks
                # when a pose is detected.
                assert len(core_landmarks) == 33, (
                    f"Expected 33 landmarks, got {len(core_landmarks)}"
                )

                status_text = f"Pose detected - {len(core_landmarks)}/33 landmarks"
                cv2.putText(
                    frame, status_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
                )

                # Print sample data periodically (not every frame, to
                # avoid flooding the console at ~30 prints/second).
                now = time.time()
                if now - last_print_time >= PRINT_INTERVAL_SECONDS:
                    last_print_time = now
                    print(f"[Core-compatible] total landmarks: {len(core_landmarks)}")
                    print(f"[Core-compatible] landmark[0] (NOSE) as (x, y, z): {core_landmarks[0]}")
                    print(f"[Detailed]         landmark[0] (NOSE) full info: {details[0]}")
                    print("-" * 60)
            else:
                cv2.putText(
                    frame, "No pose detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
                )

            cv2.imshow(WINDOW_NAME, frame)

            # waitKey(1) keeps the video loop near real-time while still
            # checking for the 'q' keypress once per frame.
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\n'q' pressed. Shutting down.")
                break

    cap.release()
    cv2.destroyAllWindows()
    print("Webcam released and OpenCV windows closed. PoseDetector closed.")


if __name__ == "__main__":
    main()
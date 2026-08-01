import cv2


class Drawer:
    """
    Draws UI information on top of the webcam frame.

    NOTE:
    Skeleton drawing is already handled by PoseDetector.
    This class only draws the information panel.
    """

    def draw_information(
        self,
        frame,
        exercise,
        reps,
        stage,
        progress,
        feedback,
        fps,
    ):

        panel_height = 220

        cv2.rectangle(
            frame,
            (0, 0),
            (380, panel_height),
            (40, 40, 40),
            -1
        )

        cv2.putText(
            frame,
            f"Exercise : {exercise}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            frame,
            f"Reps : {reps}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        cv2.putText(
            frame,
            f"Stage : {stage}",
            (20, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )

        cv2.putText(
            frame,
            f"Progress : {progress:.1f}%",
            (20, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 200, 0),
            2,
        )

        cv2.putText(
            frame,
            f"Feedback : {feedback}",
            (20, 175),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 255),
            2,
        )

        cv2.putText(
            frame,
            f"FPS : {fps:.2f}",
            (20, 210),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
        )

        return frame
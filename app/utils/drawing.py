"""
===========================================
Overlay Drawing Utility
-------------------------------------------
Draws the session information panel on top
of the webcam frame.
===========================================

NOTE:
Skeleton drawing is already handled by PoseDetector.
This class only draws the information panel.
"""

from __future__ import annotations

from typing import Optional, Sequence

import cv2

# Panel geometry
PANEL_WIDTH = 430
PANEL_BASE_HEIGHT = 250
PANEL_LINE_HEIGHT = 26
PANEL_BOTTOM_PADDING = 10

# BGR colours
COLOR_PANEL = (40, 40, 40)
COLOR_WHITE = (255, 255, 255)
COLOR_REPS = (0, 255, 0)
COLOR_STAGE = (0, 255, 255)
COLOR_PROGRESS = (255, 200, 0)
COLOR_FEEDBACK = (0, 0, 255)
COLOR_FPS = (255, 255, 0)
COLOR_GOOD = (0, 220, 0)
COLOR_BAD = (0, 165, 255)

# Progress bar geometry
BAR_ORIGIN = (20, 152)
BAR_SIZE = (340, 18)

FONT = cv2.FONT_HERSHEY_SIMPLEX
MAX_FEEDBACK_CHARS = 34


class Drawer:
    """
    Draws UI information on top of the webcam frame.

    `draw_information` keeps its original positional/keyword contract;
    `posture_feedback` and `posture_ok` are optional additions used by
    run.py to surface PostureRules output.
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
        posture_feedback: Optional[Sequence[str]] = None,
        posture_ok: Optional[bool] = None,
    ):
        """
        Draw the session panel onto `frame` and return it.

        Args:
            frame: BGR image to draw on (modified in place and returned).
            exercise: Exercise name to display.
            reps: Repetition count.
            stage: Current exercise stage, or None.
            progress: Percentage 0-100, or None when unavailable.
            feedback: Primary feedback sentence.
            fps: Frames per second.
            posture_feedback: Optional posture messages from PostureRules.
            posture_ok: Optional overall posture verdict, used to colour
                the posture block.

        Returns:
            The same frame, with the panel drawn on it.
        """
        posture_lines = list(posture_feedback or [])
        # +1 row for the posture heading itself, so the block never
        # overflows the panel background.
        panel_height = (
            PANEL_BASE_HEIGHT
            + PANEL_LINE_HEIGHT * (len(posture_lines) + 1)
            + PANEL_BOTTOM_PADDING
        )

        cv2.rectangle(frame, (0, 0), (PANEL_WIDTH, panel_height), COLOR_PANEL, -1)

        progress_value = 0.0 if progress is None else float(progress)

        cv2.putText(frame, f"Exercise : {exercise}", (20, 35), FONT, 0.7, COLOR_WHITE, 2)
        cv2.putText(frame, f"Reps : {reps}", (20, 70), FONT, 0.7, COLOR_REPS, 2)
        cv2.putText(frame, f"Stage : {stage}", (20, 105), FONT, 0.7, COLOR_STAGE, 2)
        cv2.putText(
            frame,
            f"Progress : {progress_value:.1f}%",
            (20, 140),
            FONT,
            0.7,
            COLOR_PROGRESS,
            2,
        )

        self._draw_progress_bar(frame, progress_value)

        cv2.putText(
            frame,
            f"Feedback : {self._truncate(feedback)}",
            (20, 205),
            FONT,
            0.65,
            COLOR_FEEDBACK,
            2,
        )
        cv2.putText(frame, f"FPS : {fps:.2f}", (20, 240), FONT, 0.7, COLOR_FPS, 2)

        self._draw_posture_block(frame, posture_lines, posture_ok)

        return frame

    def _draw_progress_bar(self, frame, progress_value: float) -> None:
        """Draw the repetition-progress bar for the current rep."""
        x, y = BAR_ORIGIN
        width, height = BAR_SIZE

        filled = int(width * max(0.0, min(100.0, progress_value)) / 100.0)

        cv2.rectangle(frame, (x, y), (x + width, y + height), COLOR_WHITE, 1)
        if filled > 0:
            cv2.rectangle(frame, (x, y), (x + filled, y + height), COLOR_PROGRESS, -1)

    def _draw_posture_block(
        self, frame, posture_lines: Sequence[str], posture_ok: Optional[bool]
    ) -> None:
        """Draw the optional PostureRules message list under the main panel."""
        if not posture_lines:
            return

        colour = COLOR_GOOD if posture_ok else COLOR_BAD
        # PostureRules.evaluate() returns a message per check -- passing and
        # failing alike -- so the heading states the verdict rather than
        # implying every line below it needs fixing.
        heading = "Posture OK" if posture_ok else "Posture needs attention:"
        cv2.putText(frame, heading, (20, 272), FONT, 0.6, colour, 2)

        for offset, message in enumerate(posture_lines):
            y = 272 + PANEL_LINE_HEIGHT * (offset + 1)
            cv2.putText(frame, f"- {self._truncate(message)}", (20, y), FONT, 0.55, colour, 1)

    @staticmethod
    def _truncate(text) -> str:
        """Keep overlay text inside the panel width."""
        value = str(text)
        if len(value) <= MAX_FEEDBACK_CHARS:
            return value
        return value[: MAX_FEEDBACK_CHARS - 3] + "..."

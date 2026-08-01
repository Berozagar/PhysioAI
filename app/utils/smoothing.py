"""
===========================================
Landmark Smoothing Utility
-------------------------------------------
Uses Exponential Moving Average (EMA)
to reduce landmark jitter.
===========================================
"""


class LandmarkSmoother:

    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.previous = None

    def smooth(self, landmarks):

        if landmarks is None:
            return None

        # First frame
        if self.previous is None:
            self.previous = landmarks
            return landmarks

        smoothed = []

        for prev, curr in zip(self.previous, landmarks):

            x = self.alpha * curr.x + (1 - self.alpha) * prev.x
            y = self.alpha * curr.y + (1 - self.alpha) * prev.y
            z = self.alpha * curr.z + (1 - self.alpha) * prev.z

            curr.x = x
            curr.y = y
            curr.z = z

            smoothed.append(curr)

        self.previous = smoothed

        return smoothed
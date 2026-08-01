"""
===========================================
FPS Utility
-------------------------------------------
Calculates Frames Per Second (FPS)
for real-time video processing.
===========================================
"""

import time


class FPSCounter:

    def __init__(self):
        self.previous_time = time.time()
        self.current_time = time.time()
        self.fps = 0.0

    def update(self):
        self.current_time = time.time()

        elapsed = self.current_time - self.previous_time

        if elapsed > 0:
            self.fps = 1 / elapsed

        self.previous_time = self.current_time

        return round(self.fps, 2)
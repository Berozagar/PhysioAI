import time

from app.utils.fps import FPSCounter

fps_counter = FPSCounter()

for i in range(10):
    time.sleep(0.1)

    fps = fps_counter.update()

    print(f"Frame {i+1}: FPS = {fps}")
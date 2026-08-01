import cv2
import numpy as np

from app.utils.drawing import Drawer

drawer = Drawer()

frame = np.zeros((720, 1280, 3), dtype=np.uint8)

frame = drawer.draw_information(
    frame=frame,
    exercise="Shoulder Raise",
    reps=12,
    stage="UP",
    progress=84.3,
    feedback="Raise your arm higher",
    fps=29.5,
)

cv2.imshow("Drawing Test", frame)

cv2.waitKey(0)
cv2.destroyAllWindows()
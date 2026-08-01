from app.utils.smoothing import LandmarkSmoother


class Point:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


smoother = LandmarkSmoother(alpha=0.5)

frame1 = [Point(10, 20, 0)]
frame2 = [Point(20, 30, 0)]

print("Frame 1")

result = smoother.smooth(frame1)

print(result[0].x, result[0].y)

print("Frame 2")

result = smoother.smooth(frame2)

print(result[0].x, result[0].y)
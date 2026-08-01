"""
===========================================================
Exercise Rules Module
-----------------------------------------------------------
This module defines:
1. Supported physiotherapy exercises
2. Exercise-specific angle thresholds
3. Repetition counting logic
4. Exercise progress calculation
===========================================================
"""

from dataclasses import dataclass


# ===========================================================
# Exercise Configuration
# ===========================================================

EXERCISES = {

    "shoulder_raise": {
        "joint": "shoulder",
        "start_angle": 30,
        "end_angle": 160,
        "start_stage": "down",
        "end_stage": "up",
        "feedback": {
            "low": "Raise your arm higher",
            "good": "Perfect!",
            "high": "Do not overextend your shoulder"
        }
    },

    "squat": {
        "joint": "knee",
        "start_angle": 170,
        "end_angle": 90,
        "start_stage": "up",
        "end_stage": "down",
        "feedback": {
            "low": "Go lower",
            "good": "Nice squat!",
            "high": "Stand completely upright"
        }
    },

    "knee_bend": {
        "joint": "knee",
        "start_angle": 170,
        "end_angle": 80,
        "start_stage": "straight",
        "end_stage": "bend",
        "feedback": {
            "low": "Bend your knee more",
            "good": "Excellent!",
            "high": "Do not over bend"
        }
    }

}


# ===========================================================
# Exercise State
# ===========================================================

@dataclass
class ExerciseState:
    counter: int = 0
    stage: str = None


# ===========================================================
# Exercise Rule Class
# ===========================================================

class ExerciseRule:

    def __init__(self, exercise_name: str):

        if exercise_name not in EXERCISES:
            raise ValueError(
                f"Exercise '{exercise_name}' is not supported."
            )

        self.exercise_name = exercise_name
        self.rule = EXERCISES[exercise_name]

        self.state = ExerciseState()


    # -------------------------------------------------------
    # Reset Counter
    # -------------------------------------------------------

    def reset(self):
        self.state.counter = 0
        self.state.stage = None


    # -------------------------------------------------------
    # Get Current Count
    # -------------------------------------------------------

    def get_count(self):
        return self.state.counter


    # -------------------------------------------------------
    # Get Current Stage
    # -------------------------------------------------------

    def get_stage(self):
        return self.state.stage


    # -------------------------------------------------------
    # Progress Percentage
    # -------------------------------------------------------

    def get_progress(self, angle):

        start = self.rule["start_angle"]
        end = self.rule["end_angle"]

        progress = (angle - start) / (end - start)

        progress = max(0.0, min(progress, 1.0))

        return round(progress * 100, 2)


    # -------------------------------------------------------
    # Repetition Counter
    # -------------------------------------------------------

    def update(self, angle):

        start = self.rule["start_angle"]
        end = self.rule["end_angle"]

        start_stage = self.rule["start_stage"]
        end_stage = self.rule["end_stage"]

        # Exercises where angle increases
        if start < end:

            if angle >= end:
                self.state.stage = end_stage

            if angle <= start and self.state.stage == end_stage:
                self.state.counter += 1
                self.state.stage = start_stage

        # Exercises where angle decreases
        else:

            if angle <= end:
                self.state.stage = end_stage

            if angle >= start and self.state.stage == end_stage:
                self.state.counter += 1
                self.state.stage = start_stage

        return self.state.counter, self.state.stage


    # -------------------------------------------------------
    # Exercise Feedback
    # -------------------------------------------------------

    def get_feedback(self, angle):

        start = self.rule["start_angle"]
        end = self.rule["end_angle"]

        feedback = self.rule["feedback"]

        tolerance = 10

        if start < end:

            if angle < end - tolerance:
                return feedback["low"]

            elif angle > end + tolerance:
                return feedback["high"]

            else:
                return feedback["good"]

        else:

            if angle > end + tolerance:
                return feedback["low"]

            elif angle < end - tolerance:
                return feedback["high"]

            else:
                return feedback["good"]


    # -------------------------------------------------------
    # Complete Exercise Information
    # -------------------------------------------------------

    def get_status(self, angle):

        counter, stage = self.update(angle)

        return {
            "exercise": self.exercise_name,
            "counter": counter,
            "stage": stage,
            "progress": self.get_progress(angle),
            "feedback": self.get_feedback(angle)
        }
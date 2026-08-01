"""
===========================================================
Posture Rules Module
-----------------------------------------------------------
This module validates posture for different physiotherapy
exercises using calculated joint angles.
===========================================================
"""


class PostureRules:

    def __init__(self):
        pass

    # -------------------------------------------------------
    # Shoulder Raise
    # -------------------------------------------------------
    def check_shoulder_raise(self, shoulder_angle):

        if shoulder_angle < 150:
            return False, "Raise your arm higher."

        elif shoulder_angle > 175:
            return False, "Do not overextend your shoulder."

        return True, "Perfect shoulder position."

    # -------------------------------------------------------
    # Squat
    # -------------------------------------------------------
    def check_squat(self, knee_angle):

        if knee_angle > 120:
            return False, "Go lower."

        elif knee_angle < 70:
            return False, "Do not squat too low."

        return True, "Good squat."

    # -------------------------------------------------------
    # Knee Bend
    # -------------------------------------------------------
    def check_knee_bend(self, knee_angle):

        if knee_angle > 100:
            return False, "Bend your knee more."

        elif knee_angle < 60:
            return False, "Do not over bend."

        return True, "Perfect knee bend."

    # -------------------------------------------------------
    # Back Alignment
    # -------------------------------------------------------
    def check_back_alignment(self, back_angle):

        if back_angle < 160:
            return False, "Straighten your back."

        return True, "Back posture correct."

    # -------------------------------------------------------
    # Head Alignment
    # -------------------------------------------------------
    def check_head_alignment(self, neck_angle):

        if neck_angle < 150:
            return False, "Keep your head straight."

        return True, "Head posture correct."

    # -------------------------------------------------------
    # Overall Evaluation
    # -------------------------------------------------------
    def evaluate(self, exercise, angles):

        feedback = []
        correct = True

        if exercise == "shoulder_raise":

            status, msg = self.check_shoulder_raise(
                angles["shoulder"]
            )
            feedback.append(msg)
            correct &= status

            status, msg = self.check_back_alignment(
                angles["back"]
            )
            feedback.append(msg)
            correct &= status

            status, msg = self.check_head_alignment(
                angles["neck"]
            )
            feedback.append(msg)
            correct &= status

        elif exercise == "squat":

            status, msg = self.check_squat(
                angles["knee"]
            )
            feedback.append(msg)
            correct &= status

            status, msg = self.check_back_alignment(
                angles["back"]
            )
            feedback.append(msg)
            correct &= status

        elif exercise == "knee_bend":

            status, msg = self.check_knee_bend(
                angles["knee"]
            )
            feedback.append(msg)
            correct &= status

        else:

            return {
                "correct": False,
                "feedback": ["Unsupported exercise."]
            }

        return {
            "correct": correct,
            "feedback": feedback
        }
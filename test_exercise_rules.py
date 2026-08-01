from app.models.exercise_rules import ExerciseRule

exercise = ExerciseRule("shoulder_raise")

angles = [25, 60, 90, 140, 165, 150, 100, 40, 25]

for angle in angles:
    result = exercise.get_status(angle)
    print(result)
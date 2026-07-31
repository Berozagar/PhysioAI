from models.posture_rules import PostureRules

rules = PostureRules()

angles = {
    "shoulder": 145,
    "back": 165,
    "neck": 160
}

result = rules.evaluate(
    "shoulder_raise",
    angles
)

print(result)
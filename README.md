# PhysioAI

Real-time physiotherapy exercise coach. Tracks your body through a webcam,
counts repetitions, checks your form, and gives spoken feedback as you move.

Built on MediaPipe Pose (33-point BlazePose landmarks) and OpenCV.

---

## Pipeline

```
VideoCapture ──> PoseDetector ──> LandmarkSmoother ──> MotionAnalyzer
                                                             │
                                       ExerciseRule ◄────────┤
                                       PostureRules ◄────────┘
                                             │
                               Drawer overlay + AudioFeedback
```

| Layer | Module | Responsibility |
|---|---|---|
| Input | `app/detectors/pose_detector.py` | Webcam frame → 33 pose landmarks (MediaPipe Tasks API) |
| Filter | `app/utils/smoothing.py` | Exponential moving average to remove landmark jitter |
| Analysis | `app/core/kinematics.py` | Joint angles, velocity, acceleration, range of motion, smoothness |
| Analysis | `app/core/motion_analyzer.py` | Rolling per-frame analysis; delegates reps to the exercise rule |
| Rules | `app/models/exercise_rules.py` | Rep counting + stage machine per exercise |
| Rules | `app/models/posture_rules.py` | Form checks (joint ranges, back and head alignment) |
| Output | `app/utils/drawing.py` | On-screen information panel |
| Output | `app/core/audio_feedback.py` | Non-blocking text-to-speech + sound effects |
| Entry | `run.py` | Wires all of the above into one loop |

---

## Setup

Requires **Python 3.12** (widest wheel coverage for MediaPipe and OpenCV).

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
python tools/make_sounds.py     # generate assets/sounds/*.wav
```

The pose model (`models/pose_landmarker.task`) ships with the repository.
If it is ever missing, `PoseDetector` prints the exact download command.

> **VS Code:** run *Python: Select Interpreter* and pick `.venv` — otherwise
> the editor reports the dependencies as missing even though they are installed.

---

## Running

```bash
python run.py                          # webcam 0, shoulder_raise
python run.py --exercise squat
python run.py --exercise knee_bend
python run.py --source clip.mp4        # analyse a recorded video
python run.py --self-test              # full pipeline, no camera needed
python run.py --list-exercises
python run.py --list-cameras           # which webcam indices actually work
```

### Setting up a webcam

`--list-cameras` probes indices 0–5 across every capture backend and reports
which ones deliver frames:

```
$ python run.py --list-cameras
camera 0  backend=DirectShow   use: python run.py --source 0
```

On Windows the capture layer tries **DirectShow → Media Foundation → default**,
because the OpenCV default (Media Foundation) is often slow to open or fails
on webcams DirectShow handles fine. Each candidate is confirmed by actually
decoding a frame — `isOpened()` alone is not trusted, since some drivers
report success and then fail every read.

### Controls

| Key | Action |
|---|---|
| `q` / `Esc` | Quit |
| `r` | Reset repetition count and history |
| `m` | Mute audio |

### Useful flags

| Flag | Purpose |
|---|---|
| `--source` | Webcam index (`0`) or path to a video file |
| `--smoothing` | EMA alpha in `(0, 1]`; `0` disables smoothing (default `0.3`) |
| `--no-audio` / `--no-voice` / `--no-sfx` | Disable audio, speech only, or effects only |
| `--headless` | Run without opening a window (CI, no display) |
| `--max-frames N` | Stop after N frames |
| `--save-frame PATH` | Write the last annotated frame to an image |
| `--log-level DEBUG` | Verbose internal logging |

### No webcam?

`--self-test` drives the entire analysis stack with synthetic 33-point
landmarks — no camera, no window — and verifies the repetition count is
exact. It is the quickest way to confirm an install is healthy:

```bash
python run.py --self-test --exercise squat --self-test-reps 3
```

---

## Supported exercises

| Exercise | Joint | Range | Counted when |
|---|---|---|---|
| `shoulder_raise` | shoulder | 30° → 160° | Arm raised past 160° then lowered below 30° |
| `squat` | knee | 170° → 90° | Knee bent past 90° then straightened past 170° |
| `knee_bend` | knee | 170° → 80° | Knee bent past 80° then straightened past 170° |

### Adding an exercise

Adding an entry to `EXERCISES` in `app/models/exercise_rules.py` is enough
to get rep counting, staging, progress and `--self-test` working — those
are all derived from the rule's own thresholds.

**Posture checks are not automatic.** `PostureRules.evaluate()` branches on
hardcoded exercise names, so a new exercise also needs:

1. a check method + an `elif` branch in `app/models/posture_rules.py`, and
2. an entry in `REQUIRED_POSTURE_JOINTS` in `run.py`.

Without those two, the exercise runs fine but silently performs no form
checking (`run.py` logs a warning at startup).

---

## A note on smoothing

`LandmarkSmoother` lags a moving signal by roughly `(1 - alpha) / alpha`
frames. At the default `alpha=0.3` that is about 2.3 frames (~78 ms at
30 FPS), which is invisible in normal use. But it does clip the peak of
very fast movements, which can stop a rep from registering. If reps are
being missed on quick movements, raise `--smoothing` toward `1.0` (less
smoothing, more jitter) or move more deliberately.

---

## Tests

```bash
pip install -r requirements-dev.txt
pytest                                          # 210 tests, ~2s
pytest --cov=app --cov=run --cov-report=term-missing
```

Currently **81% coverage**. The pure logic is fully covered (kinematics 99%,
exercise/posture rules 100%, smoothing 100%, drawing 100%, motion analyzer 96%).
The uncovered remainder is the hardware layer — `pose_detector.py` needs a
camera and the model, and `run_live()` needs a live capture device.

`pytest.ini` restricts collection to `tests/` **on purpose**: the root-level
`test_*.py` files are manual demo scripts, not pytest tests, and
`test_drawing.py` would block an automated run on a GUI window.

### Demo scripts (root)

These print results for eyeballing rather than asserting.

| Script | Needs |
|---|---|
| `test_core.py` | Nothing — full core integration check |
| `test_exercise_rules.py`, `test_posture_rules.py`, `test_smoothing.py`, `test_fps.py` | Nothing |
| `test_pose_detector.py` | A webcam + display |
| `test_drawing.py` | A display |

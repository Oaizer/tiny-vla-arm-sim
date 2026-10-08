# TinyVLA for a simulated robot arm

This project trains a small vision and language policy to control a simulated Panda arm. It runs on a CPU in Linux or Ubuntu under WSL. The second version adds shape recognition, three target pads, placing and lifting tasks, varied scenes, and a feedback controller.

## Results

The trained model has **17,896 parameters**. Its checkpoint is **78,815 bytes**. We tested two controllers in 800 trials across ten test sets. The feedback controller succeeded in 39 of 40 standard trials, 32 of 40 trials where an object moved, and 24 of 40 trials with a forced drop. It stopped before moving in 39 of 40 trials where the requested object was missing. Feedback did worse than one-shot control in some changed scenes.

Read the [full results](artifacts/RESULTS.md). Watch the [moved-object demo](artifacts/demo.mp4) or the [slip recovery demo](artifacts/slip-demo.mp4). The feedback rules are hand-written. The model learns to read images and instructions and to predict arm waypoints.

## Run the project

You need Linux or Ubuntu under WSL, Python **3.11.16**, and [uv](https://docs.astral.sh/uv/). The lock file pins each package version and hash. It uses the CPU build of PyTorch.

```bash
bash run.sh
```

This command checks the project, creates a new run folder, generates training and validation data, trains the model, and tests it. It also saves a demo, videos of failed trials, the results, and a file of hashes. It keeps earlier runs. You do not need a GPU or a paid service.

To test an existing model or make a demo:

```bash
uv run --locked tinyvla evaluate --run-dir artifacts --suites nominal object_moved slip
uv run --locked tinyvla demo --run-dir artifacts \
  --instruction 'put the red cylinder on the cyan pad' --event object_moved
```

The included experiment and its files are in `artifacts/`. New runs go to `artifacts/runs/<timestamp>/`.

## Project files

- `src/tinyvla/` contains the language rules, simulator, model, training code, controller, tests, and command line tool.
- `configs/project.json` sets the model size, data, training, and test limits.
- `tests/` checks the supported language, repeatable simulator resets, and safe recovery rules.
- `docs/MODEL_GOAL.md` describes the model and its limits.
- `docs/BENCHMARK.md` describes the scene changes and how we score each trial.
- `baseline-v1/` contains the first version and its report.
- `artifacts/` contains the trained model, data, logs, videos, and results.

## What the model can do

It can identify a red cube, red cylinder, blue cube, or green cylinder. It can place or lift one of these objects. Place commands can name a yellow, purple, or cyan pad. The camera calibration is supplied to the model.

The model only understands the words and sentence forms listed in [the model guide](docs/MODEL_GOAL.md). It does not understand arbitrary language or unfamiliar objects. The camera view is blocked during a grasp, so the controller looks again between steps. It does not track objects through the blockage. We have only tested this model in simulation.

The model predicts image locations, object presence, which object and task the words describe, and arm waypoints. Standard inverse kinematics and joint controls move the arm. The hand-written feedback controller looks at a new camera image between steps. It checks the finger sensors after a grasp, retries a limited number of times, and stops when its attempt limit is reached. It cannot read object labels or the simulator's scoring data.

## Change or test the code

```bash
uv sync --locked
uv run --locked python -m unittest discover -s tests -v
```

The GitHub Actions workflow runs the project checks. The Git repository includes the experiment data and model. The PyBullet arm and other packages keep their own licenses.

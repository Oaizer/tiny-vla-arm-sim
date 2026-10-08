# TinyVLA arm project

A reproducible CPU project for learning vision-and-language-conditioned actions on a simulated Panda arm. Version 2 expands the original prototype to shape discrimination, three destinations, placing/lifting tasks, randomized scenes, and bounded visual feedback/recovery.

## Run

Linux or Ubuntu under WSL, Python **3.11.16**, and [uv](https://docs.astral.sh/uv/). Dependencies and their hashes are pinned in `uv.lock`; the PyTorch CPU index is explicit.

```bash
bash run.sh
```

This one command checks simulator/language contracts, creates a fresh experiment directory, generates training/validation data, trains, refits the action heads using training data only, runs paired benchmark suites, records a demo, and writes results plus an integrity manifest. It does not overwrite a previous experiment. No GPU or paid service is needed.

For an existing experiment:

```bash
uv run --locked tinyvla evaluate --run-dir artifacts --suites nominal object_moved slip
uv run --locked tinyvla demo --run-dir artifacts \
  --instruction 'put the red cylinder on the cyan pad' --event object_moved
```

The completed local experiment is stored under `artifacts/` in the delivered folder/archive. A fresh `run.sh` run uses `artifacts/runs/<timestamp>/`. On the current machine, the already-installed runtime also runs commands with `/var/tmp/tiny-vla-env/bin/python -m tinyvla.cli` instead of `uv run --locked tinyvla`.

## Project layout

- `src/tinyvla/`: language contract, simulator, learned model, data, training, feedback controller, evaluation, and CLI.
- `configs/project.json`: model budgets, sample counts, seed splits, training parameters, and benchmark/controller settings.
- `tests/`: contracts for language, deterministic resets, calibrated coordinates, and bounded recovery behavior.
- `docs/MODEL_GOAL.md`: architecture decision and supported language.
- `docs/BENCHMARK.md`: variation ranges, perturbations, scoring, and comparison protocol.
- `baseline-v1/`: original source and summary for reference.
- `artifacts/`: generated data, checkpoints, logs, videos, reports, and manifests; excluded from Git.

## What learns and what is scripted

The neural model learns image locations/presence, instruction selection, and Cartesian/gripper waypoint heads. A known camera calibration maps learned image positions into the tabletop frame. Conventional inverse kinematics executes the commands. A scripted supervisor replans from RGB at phase boundaries and uses finger encoders to detect failed grips and retry. It never accesses object labels or the scorer.

Supported objects are red cube, red cylinder, blue cube, and green cylinder. Pad colors are yellow, purple, and cyan. Commands use a declared vocabulary with placing and lifting verbs. This compact system does not provide unrestricted language, unfamiliar object recognition, continuous visual tracking through occlusion, or physical-arm validation.

All failed trials and stop reasons are retained. Reported metrics distinguish the expanded v2 task distribution from the original easier v1 baseline. The recovery benchmark compares two controllers using the same v2 model and matching scenes.

## Development

```bash
uv sync --locked
uv run --locked python -m unittest discover -s tests -v
```

Git history is local. The included GitHub Actions workflow can run the contract checks when this repository is later hosted; no repository has been published by this work.

Robot assets and the physics engine come from [PyBullet](https://github.com/bulletphysics/bullet3). Their upstream asset licenses apply. PyTorch, NumPy, Pillow, and ImageIO retain their respective licenses.

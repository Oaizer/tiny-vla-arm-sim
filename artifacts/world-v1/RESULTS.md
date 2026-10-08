# Simulation world model results

This experiment uses only PyBullet data. The model predicts what will happen after an arm command. A short planner uses those predictions to compare nearby grasp choices.

## Data

| Split | Episodes | Executed commands | Seed range |
|---|---:|---:|---|
| train | 800 | 45429 | 100000–100799 |
| validation | 100 | 5155 | 110000–110099 |
| test | 100 | 5737 | 120000–120099 |

Each record contains the camera image, sensor features, actual command, command duration, and the next observation. Three observations form the model input. Episodes include varied contact heights, poor approaches, fallen cylinders, drops, and retries. Training labels include simulated object poses. The planner receives only camera, calibration, and arm sensor inputs.

## Model

The original visual network stays fixed. It reads a 64 × 64 image resized to its original input size. A small encoder reads its spatial features and three recent sensor observations. The state has 71 learned values and 25 direct landmark and sensor values. Three recurrent predictors estimate future states from commands and their durations. A residual readout estimates object pose, workspace presence, held-object score, and arm sensor values.

We first train the feature encoder on simulation labels. We then freeze it and train future prediction. In the JEPA phase, we train that encoder with future-feature loss and a slowly updated target encoder. A feature-variance loss and the label losses help prevent a constant encoding. The visual CNN stays fixed in both phases. This is a small JEPA-style hybrid with simulation labels, not a reproduction of V-JEPA or a fully self-supervised model.

| Model | Parameters, excluding target encoder | Checkpoint bytes |
|---|---:|---:|
| fixed | 537,625 | 3,896,941 |
| jepa | 537,625 | 3,896,901 |

The fixed visual network is additional to these parameter counts. The target encoder is used during training and is saved in the checkpoint. It is not used by the planner.

## Prediction tests

Errors below are measured on held-out episodes. Position error is RMSE over x, y, and z for objects within the supported workspace: x 0.25–0.80 m, y −0.40–0.40 m, z 0–0.40 m. Some failures throw objects outside the view. Their raw poses stay in the dataset, and the JSON report retains unbounded pose error. Their workspace-presence labels still contribute to training. Many objects remain still. The moving-object measure uses objects within the workspace whose per-axis RMS movement exceeds 1 cm. A lower feature error alone does not prove better robot control.

| Model / input | Position error after 1 command, mm | After 3 commands, mm | Moving objects after 3 commands, mm |
|---|---:|---:|---:|
| fixed / normal | 57.17 | 57.43 | 94.64 |
| fixed / shuffled actions | 57.30 | 58.14 | 99.02 |
| fixed / hold current visual estimate | 57.99 | 59.53 | — |
| jepa / normal | 57.13 | 57.22 | 91.61 |
| jepa / shuffled actions | 57.50 | 58.65 | 99.60 |
| jepa / hold current visual estimate | 57.92 | 59.32 | — |

For reference, holding the true current object position gives 6.34 mm after one command and 14.96 mm after three. That reference uses simulator state and is not available to the controller.

The JSON results also include feature error, a no-change feature prediction, held-object score error, sensor error, sample counts, and checkpoint hashes. Shuffled-action tests keep the observations but assign actions from other samples.

## Arm tests

Both controllers use the same visual policy, servo controls, scene seeds, and task rules. The first attempt uses the existing controller. After a failed grasp, the planner can change nearby grasp commands. It compares short sequences, executes the first command, and observes again. A penalty discourages large changes and disagreement among the three predictors. Their disagreement is a heuristic, not a calibrated confidence bound.

| Scene | Existing feedback success | Fixed-encoder planner | JEPA planner |
|---|---:|---:|---:|
| Nominal | 38/40 | 38/40 | 38/40 |
| Tipped | 22/40 | 19/40 | 20/40 |
| Slip | 20/40 | 19/40 | 19/40 |
| Object moved | 37/40 | 37/40 | 37/40 |
| Dynamics | 38/40 | 38/40 | 38/40 |
| Missing object: stop before movement | 39/40 | 39/40 | 39/40 |

Across 200 task trials per controller, the existing controller succeeded 155 times, the fixed-encoder planner 151 times, and the JEPA planner 152 times. Missing-object stops are reported separately.

The fixed planner gained 1 successes and lost 5 successes on matched trials.

The jepa planner gained 2 successes and lost 5 successes on matched trials.

This run does not show an overall success gain from the JEPA planner. The existing feedback controller remains the default. The world model and planner are available as an experiment.

The largest per-suite p95 planner call took 4.10 ms. This covers candidate prediction and selection. Camera capture and feature extraction are separate; their largest per-suite p95 was 33.60 ms. These timings come from this CPU run.

Each JSONL file contains all trials, changes to commands, stop reasons, retries, and timing. The first trial in each pair has a video. Confidence intervals and results by object are in the JSON summaries. Forty trials per condition give limited precision.

## Limits

The model has a short history and a short prediction horizon. The gripper keeps its original orientation. A blocked camera view can hide the object pose. The model can favor a plan that only looks good in its own predictions. It cannot anticipate an external object move before observing it. The planner still relies on scripted task phases and retry rules. These results apply to simulation only.

## Repeat the run

From the project folder, run:

```bash
uv run --locked python -m tinyvla.world_cli run
```

The command collects data, trains both stages, measures prediction, compares controllers, and writes this report. Run each stage separately with `collect`, `train`, `predict`, `evaluate`, or `report`. Use `--development` for development scenes. Training and evaluation write their run files, so use a new `--root` to keep an earlier run.

Large trajectory files stay in the local output folder. The repository contains the source, settings, trained models, results, hashes, and small videos. `manifest.json` records the exact files for this run.

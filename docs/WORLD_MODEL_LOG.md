# World model experiment log

## Plan and data

We used the existing PyBullet arm, trained VLA, and CPU environment. We first recorded 200 pilot episodes. We then extended the dataset to 800 training episodes, 100 validation episodes, and 100 prediction-test episodes. Collection used separate seed ranges. All data is simulated.

Each transition stores a completed command and its duration. This is a change from the initial idea of fixed-rate video steps. It matches the existing servo API and keeps the original controller available for matched tests. The recorder saves actual images and sensor values before and after motion.

## First fit

The first fit tried to predict every object's full position. Validation position error was about 235 mm. An inspection of validation labels found positions from x −4.07 m to x 5.13 m. Some bad grasps had thrown objects far outside the camera view. These failures were valid simulation outcomes, but their exact positions were not observable.

We kept the raw records. We added a derived workspace-presence label and applied pose losses only within the supported workspace. The report also keeps unbounded error. We added explicit landmark and sensor coordinates to the latent state, with a learned residual readout. The first fit's weights and logs remain in the local `attempts/initial_model` folder. That older weight layout needs the earlier architecture and is not a deployment checkpoint.

## Final model training

The fixed CNN supplies spatial features. The trainable state combines 25 direct input values and 71 learned values. Three recurrent predictors learn future states. Training uses a supervised warmup, a fixed-encoder stage, and a JEPA stage with a slowly updated target encoder. Both stages use the same trajectories and simulation labels. They are selected on validation position error.

Prediction tests use 100 separate episodes. We compare actual commands, shuffled commands, and no-change predictions. The learned latent standard deviation is also saved to check for a constant representation. Model weights are held fixed during controller development and the final arm tests.

## Planner development

The first change penalty was too strong. The planner kept the baseline command throughout the completed development trials. Those results are retained in `development/`.

The next setting reduces the change penalty and allows planning only after a failed grasp. This focuses the experiment on recovery. The new development run is in `development-active/`. It compares both model stages on normal, tipped-object, and slip scenes. Final evaluation uses a separate set of seeds.

The early development run used four inference threads. The later run uses one. Small numerical changes can affect simulated contact outcomes. Each comparison uses the same thread count for its baseline and candidate. Timings from concurrent development runs are not used as the final speed result.

## Interface checks

All eight project checks passed before the controller comparison. The three new checks confirm recorded command timing, unchanged servo motion through the sensor wrapper, and a planner interface that works without object-state methods. Full arm outcomes and prediction errors provide the main evidence for model behavior.

## Evidence to read

- `artifacts/world-v1/RESULTS.md`: the measured result and limits.
- `prediction_results.json`: prediction errors, baselines, and model hashes.
- `model/training.jsonl`: both training stages and validation scores.
- `evaluation/`: each final trial, paired videos, and summaries.
- `manifest.json`: source, model, and trajectory hashes.
- `train_data.json`, `validation_data.json`, `test_data.json`: the episode inventory.

The report must distinguish a working world-model pipeline from a proven gain in robot success. Failed trials stay in the evidence.

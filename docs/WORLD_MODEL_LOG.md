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

On those 48 development trials, the existing controller succeeded 33 times, the fixed-encoder planner 32 times, and the JEPA planner 30 times. Both planners changed commands. Neither earned a change to the default controller. We froze these settings and included both stages in a final 720-trial comparison. `FINAL_PROTOCOL.json` records the model hashes and source commit before that comparison.

The early development run used four inference threads. The later run uses one. Small numerical changes can affect simulated contact outcomes. Each comparison uses the same thread count for its baseline and candidate. Timings from concurrent development runs are not used as the final speed result.

## Interface checks

All eight project checks passed before the controller comparison. The three new checks confirm recorded command timing, unchanged servo motion through the sensor wrapper, and a planner interface that works without object-state methods. Full arm outcomes and prediction errors provide the main evidence for model behavior.

The complete dataset has 56,321 executed commands: 45,429 in training, 5,155 in validation, and 5,737 in prediction tests. The raw trajectory files occupy about 410 MB. A separate local data ZIP contains those files. They are not added to Git.

## Evidence to read

- `artifacts/world-v1/RESULTS.md`: the measured result and limits.
- `prediction_results.json`: prediction errors, baselines, and model hashes.
- `model/training.jsonl`: both training stages and validation scores.
- `evaluation/`: each final trial, paired videos, and summaries.
- `manifest.json`: source, model, and trajectory hashes.
- `train_data.json`, `validation_data.json`, `test_data.json`: the episode inventory.

The report must distinguish a working world-model pipeline from a proven gain in robot success. Failed trials stay in the evidence.

## Second experiment: keep the grasp offset consistent

We inspected the version 1 paired outcomes. JEPA won two trials and lost five against feedback. All seven involved cylinders in tipped or slip scenes. We then replayed 25 candidate sequences from the first planning state in each of those seven trials. The selected sequence had a worse measured short-horizon score in all five old losses. Position error was 76.9 mm across the 137 candidate endpoints that stayed in the supported workspace. The repeated baseline branches reproduced their endpoints within one micrometre. These selected cases show a prediction and ranking problem; they do not estimate its frequency across all scenes.

We tested one execution change. The planner selects an offset during open descent and keeps it through closure. The normal controller then handles lift. We held the weights, candidate set, score, and recovery trigger fixed. Nine interface and project tests passed. Commit `4da4b1b` and `artifacts/world-v2/FINAL_PROTOCOL.json` froze the evaluated source and inputs before the new test.

The fresh comparison used 720 trials: three controllers, six conditions, and 40 trials per condition. Feedback completed 150 of 200 tasks. Original JEPA and the consistent-grasp variant each completed 145. Each planner had zero paired wins and five losses against feedback; the two-sided exact paired test gave p=0.0625. Each controller stopped before motion in 39 of 40 missing-object trials. Those missing-object trials are excluded from the task-success denominator.

The consistent-grasp variant reduced planning calls from 146 to 41. Mean task control time was 1.032 seconds for feedback, 1.085 for original JEPA, and 1.081 for the new variant. Fewer planning calls did not produce better task outcomes or a material measured speed gain in this run. The logged `changed_commands` field counts accepted planner choices and excludes the new variant's repeated offset at closure.

Feedback remains the default. The execution change is available only as an experimental variant. The next useful target is contact prediction and candidate ranking, using separate development scenes. Keep these final seeds for reproduction, not for choosing new settings. Read the version 2 report, prediction audit, paired trial records, and output manifest for the full evidence.

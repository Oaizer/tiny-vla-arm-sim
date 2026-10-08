# Coherent grasp experiment

All trials use PyBullet simulation. The model weights and training data are unchanged.

## Change and reason

The previous JEPA planner won two paired trials and lost five against feedback. All seven involved cylinders in tipped or slip scenes. It changed commands in 31 episodes. Many selected offsets moved the hand down by about 11 or 12 mm.

Code inspection found that descent, closure, and early lift could each get a new offset. The scored three-command sequence was therefore not an execution commitment. This is a plausible fault, not a proven cause of the seven different outcomes. The old logs do not record each proposed command or each predicted state, so they cannot establish prediction accuracy at those decisions.

The new variant chooses once during the open descent. It applies the same offset to the next closure, then returns lift control to the existing controller. A command outside this sequence clears the offset. The score, candidate set, weights, and recovery trigger remain fixed. The model still approximates the lift with a short vertical command; it does not predict the full task.

## Frozen test

The protocol was saved before evaluation. Each controller uses the same 40 seeds per condition. Six conditions and three controllers give 720 trials. The fresh seed blocks start at 170000 and increase by 1000 per condition. No parameter selection uses these results. FINAL_PROTOCOL.json records the source, input hashes, environment, and settings.

| Condition | Feedback | Original JEPA | Coherent grasp |
|---|---:|---:|---:|
| nominal | 37/40 | 37/40 | 37/40 |
| tipped | 12/40 | 12/40 | 12/40 |
| slip | 27/40 | 22/40 | 22/40 |
| object_moved | 35/40 | 35/40 | 35/40 |
| dynamics | 39/40 | 39/40 | 39/40 |
| missing | 0/40 | 0/40 | 0/40 |

The missing-object condition measures stopping, not task completion. Do not add it to the task-success denominator.

| Controller | Task success (200 trials) | Stops before motion, missing (40) | Mean control seconds, tasks | Planner calls | Accepted planner choices |
|---|---:|---:|---:|---:|---:|
| feedback | 150/200 | 39/40 | 1.032 | 0 | 0 |
| jepa | 145/200 | 39/40 | 1.085 | 146 | 137 |
| coherent | 145/200 | 39/40 | 1.081 | 41 | 37 |

## Paired outcomes

jepa: 0 wins and 5 losses against feedback; two-sided exact paired test p=0.0625. Commands changed in 39 episodes. Full seed and object records are in paired_results.json.

coherent: 0 wins and 5 losses against feedback; two-sided exact paired test p=0.0625. Commands changed in 36 episodes. Full seed and object records are in paired_results.json.

## Decision

The consistent-grasp variant completed five fewer tasks than feedback. It matched the original JEPA total of 145 successes in 200 task trials.

Keep feedback as the default. This single run and a single trained model do not establish a reliable general gain. The planner remains experimental. The offline candidate audit found errors in position and held-object prediction. Improve those predictions and check candidate ranking on separate development scenes before another confirmation run. Use new seeds for any later confirmation.

## Reproduce

Run `python -m tinyvla.world_iteration run` from the project root. Use `--root` for a new output directory. The command refuses to overwrite an existing protocol. It uses the existing version 1 checkpoints and does not retrain them. Run `python -m tinyvla.world_iteration report` to rebuild this report. Training remains available through `python -m tinyvla.world_cli run`; its data and model versions are stored in the original experiment manifest.

The interface test checks that closure retains the selected offset, lift receives the original command, and an interrupted sequence clears the offset. Trial logs add requested position, finger command, and command duration to each planner decision. Videos show the first trial in each condition; those trials are not necessarily planner interventions.

## Measurement note

The log field `changed_commands` counts accepted planner choices. The consistent-grasp variant also applies each retained offset to closure; that second application is not a new planner choice. Read [the candidate prediction audit](PREDICTION_AUDIT.md) for the separate offline prediction checks.

# Direct contact prediction and candidate ranking

This experiment improves and measures contact prediction with PyBullet labels. It does not connect the new head to the robot controller.

## Data and label

PyBullet records the normal force between the task object and either gripper finger (link 9 or 10) after each servo command. A label is positive when total finger force exceeds 0.01 N. The camera, arm features, and recorded command are the model inputs. Simulator contact force is only a target and scoring label.

| Split | Scenes | Seed range | Servo transitions | Windows | Contact rate by object: red cube, blue cube, green cylinder, red cylinder |
|---|---:|---:|---:|---:|---|
| train | 240 | 180120–180359 | 13347 | 12387 | 12.6%, 11.8%, 9.8%, 8.6% |
| validation | 80 | 181240–181319 | 4205 | 3885 | 12.7%, 12.9%, 9.2%, 9.1% |
| test | 80 | 182320–182399 | 4390 | 4070 | 12.5%, 11.7%, 10.1%, 10.6% |

The training windows, validation windows, and prediction test windows come from separate episodes. The later candidate ranking uses another 40 scenes, seeds 183500–183539.

## Contact prediction

A 6,468 parameter head predicts finger contact from each of the existing JEPA model's three future latent states. The JEPA model is frozen. The head was selected at training step 400; per-object scales and biases were fitted on validation scenes only.

| Held-out test measure | New contact head | Existing geometric held score | No-contact score |
|---|---:|---:|---:|
| Macro average precision | 0.585 | 0.223 | — |
| Macro Brier score | 0.068 | 0.138 | 0.118 |
| Macro calibration error, 10 bins | 0.041 | 0.131 | — |

Lower Brier and calibration error are better. Average precision measures how well a score finds the rarer contact steps. The no-contact baseline always predicts zero contact.

| Object | Contact rate | Head average precision | Old score average precision | Head Brier | Old score Brier | Head precision / recall at 0.5 |
|---|---:|---:|---:|---:|---:|---:|
| red cube | 13.2% | 0.481 | 0.586 | 0.087 | 0.078 | 0.529 / 0.330 |
| blue cube | 12.4% | 0.877 | 0.086 | 0.040 | 0.152 | 0.748 / 0.815 |
| green cylinder | 10.5% | 0.725 | 0.088 | 0.049 | 0.183 | 0.792 / 0.572 |
| red cylinder | 11.2% | 0.257 | 0.131 | 0.096 | 0.137 | 0.340 / 0.093 |

The head improves the macro test measures. It does not improve every object: for the red cube, its average precision and Brier score are worse than the old score. Red-cylinder recall is 9.3% at the stated threshold. The head is not ready to control grasp choices.

## Fresh candidate ranking check

We ran 40 new scenes and 1000 candidate branches. At the contact step, the top contact-head choice made finger contact in 62.5% of scenes. The unchanged command made contact in 67.5%; the old score's top choice did so in 52.5%. At least one candidate made contact in 87.5% of scenes.

Across scenes with at least one positive candidate, mean candidate average precision was 0.778 for the new head and 0.738 for the old score. Against the unchanged command, the head had 3 wins and 5 losses in the paired top-choice contact result (two-sided exact p=0.727).

| Object | Scenes | Head top choice contact | Unchanged command contact | Old score top choice contact |
|---|---:|---:|---:|---:|
| red cube | 8 | 75.0% | 75.0% | 62.5% |
| blue cube | 12 | 66.7% | 75.0% | 66.7% |
| green cylinder | 8 | 75.0% | 62.5% | 37.5% |
| red cylinder | 12 | 41.7% | 58.3% | 41.7% |

The ranking result is mixed. The contact head ranks candidate contacts slightly better overall than the old score, but the top choice underperforms the unchanged command. The differences are small in 40 scenes. This audit does not show better grasp success or task completion.

## Decision and next step

Keep the contact head offline. Do not connect it to the arm planner yet. It improves average contact prediction overall, but misses many red-cylinder contacts and does not select a better top grasp than the unchanged command. Collect contact and slip examples that focus on the weak object and improve candidate ranking on development seeds. After that, run a new, frozen controller comparison.

The first candidate-audit attempt selected only 21 of its planned 40 scenes because its selector skipped the first open approach. It is saved as an incomplete attempt and is not included in these results. The reported ranking check uses a corrected selector and a fresh seed range.

All scenes are simulated. No controller comparison or real-arm test used the new head. See `report.json`, `candidate_rank_v2.json`, `candidate_rank_v2_protocol.json`, `training.jsonl`, and `manifest.json` for the full results and hashes.

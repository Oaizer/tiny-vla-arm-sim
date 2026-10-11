# V5 offline grasp candidate reranker

## Question

The V4 contact head improved red-cylinder contact prediction, but its independent ranking audit did not beat the unchanged grasp. V5 trained a small model to rank nearby grasp branches. The arm controller and world model stayed fixed.

## Simulated data

We collected 40 training scenes, 20 validation scenes, and 40 test scenes. All used the red cylinder and separate seed ranges: 188000–188039, 188100–188119, and 188200–188239. The run created 25 candidate branches per scene. Labels use measured PyBullet finger force above 0.01 N after the approach and close commands. The high-noise probability was 65%; 80% of training scenes were planned to tip the target; 45% were planned to include a forced slip.

The reranker has 241 parameters. It scores the three predicted contact probabilities, three old geometric scores, candidate offset, predicted target offset, and tipped-scene flag. Training used the training scenes only. We selected step 100 using candidate average precision across the validation scenes.

## Fresh test

| Measure | Reranker | Contact head | Old geometric score | Unchanged grasp |
|---|---:|---:|---:|---:|
| Top-choice contact | 29/40 (72.5%) | 20/40 (50.0%) | 16/40 (40.0%) | 26/40 (65.0%) |
| Mean per-scene candidate average precision | 0.816 | 0.679 | 0.649 | 0.777 |

The reranker beat the unchanged grasp in 7 paired scenes and lost in 4; 29 scenes had the same contact outcome. This is a modest positive result on a small test set.

## Independent replication

We froze the reranker and tested it on 40 new scenes, seeds 188240–188279, with another 1,000 branches. It selected contact in 27/40 scenes (67.5%), compared with 25/40 (62.5%) for the unchanged grasp. Candidate average precision was 0.832 for the reranker and 0.801 for the unchanged grasp. The paired result was 5 wins, 3 losses, and 32 ties.

Across the two audits, the reranker selected contact in 56/80 scenes, compared with 51/80 for the unchanged grasp. It had 12 paired wins and 7 losses; a two-sided exact sign test gives p=0.359. The direction is consistent, but the evidence is not strong enough to claim a reliable control gain.

## Decision

Keep the reranker offline. It is a compact model and it improves contact candidate ranking in both fresh checks. The paired margins are small, and the tests measure contact after closure only. They do not measure lift, retention, object placement, or end-task success. A larger independent ranking test is the next step. Consider a frozen controller comparison only if that test confirms the gain.

All scenes are simulated. The model did not control the arm, and no physical robot was used. See `protocol.json`, `test_results.json`, `training.jsonl`, the replication folder, and `manifest.json` for hashes and per-scene results.

# Targeted red-cylinder contact study

## Scope

We added simulated red-cylinder contact and slip scenes after version 3 showed weak red-cylinder recall. We trained the new head on 103 targeted scenes, selected the checkpoint with 24 targeted validation scenes, and measured it on 40 fresh targeted test scenes. The version 3 model and arm controller stayed fixed. The contact head did not control the arm.

The targeted set had 8,643 training transitions, 2,059 validation transitions, and 3,236 test transitions. The training split contained 81 tipped scenes and 51 forced-slip scenes. These rates are close to the planned 80% and 45%. The validation and test splits had 20/24 and 29/40 tipped scenes, and 12/24 and 13/40 forced-slip scenes. The test seeds were 186200–186239.

## Red-cylinder test

| Measure | New targeted head | Version 3 head | Old geometric score |
|---|---:|---:|---:|
| Average precision | 0.777 | 0.732 | 0.292 |
| Brier score | 0.112 | 0.144 | 0.200 |
| Calibration error, 10 bins | 0.097 | 0.138 | 0.187 |
| Precision at 0.5 | 0.917 | 0.954 | 0.967 |
| Recall at 0.5 | 0.298 | 0.072 | 0.043 |

The new head improved average precision from 0.732 to 0.777 and recall from 7.2% to 29.8% on these fresh red-cylinder scenes. Its Brier score improved from 0.144 to 0.112. Calibration remains weak (error 0.097), and it still misses about seven in ten contact observations at the 0.5 threshold. Precision is high, but is not enough on its own to support a grasp decision.

Across the combined fresh targeted and prior held-out test episodes, the new head had macro average precision 0.568 and macro Brier score 0.058. The main targeted red-cylinder result above is the more direct test of this data change.

## Fresh candidate ranking

We ran 20 new red-cylinder scenes and 500 offline candidate branches. The new head selected a contacting candidate in 13/20 scenes (65.0%). The unchanged command made contact in 13/20 (65.0%); the old score did so in 9/20 (45.0%). At least one candidate made contact in 15/20 scenes. Candidate average precision was 0.849 for the new head and 0.769 for the old score.

This ranking result favors the new score over the old geometric score, but it ties the unchanged command's top-choice contact rate. The new head and unchanged command had the same contact outcome in all 20 scenes. The sample is small. The ranking check measures finger contact after closure, not object lift, grasp retention, or task completion.

## Independent ranking replication

We froze the head and ran a separate check on 40 new scenes (seeds 187020–187059), with 1000 candidate branches. The new head's top candidate contacted in 20/40 scenes (50.0%). The unchanged command contacted in 22/40 (55.0%); the old score contacted in 18/40 (45.0%). Candidate average precision was 0.742 for the new head and 0.719 for the old score. The new head had 3 wins and 5 losses against the unchanged command.

The replication keeps the slight candidate-ranking advantage over the old score, but the new head's top candidate contacted less often than the unchanged command. This confirms that the version 4 score does not yet provide a reliable grasp-selection gain.

## Decision

Keep the new head offline. Red-cylinder contact prediction improved on the new test scenes, but the independent ranking replication did not beat the unchanged command. Calibration remains weak. Collect more diverse grasp and slip scenes and improve the selection result before planning a controller comparison. No real robot was used.

The checkpoint was selected at training step 900; it has 6,468 parameters. Read `report.json`, both ranking result and protocol pairs, the episode metadata, and `manifest.json` for details and hashes.

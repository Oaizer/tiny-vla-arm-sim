# Candidate prediction audit

This audit uses seven old version 1 trials where JEPA and feedback had different task outcomes. It checks the first planning decision in each trial. Each of the 25 candidate sequences runs from the same saved physics state. Simulation labels are used only for this offline test. They are not given to the controller.

| Seed | Scene | Old task outcome vs feedback | Position RMSE, mm | Held-score MAE | Actual selected cost gain |
|---|---|---|---:|---:|---:|
| 152029 | slip | win | 71.2 | 0.464 | +2.580 |
| 152030 | slip | win | 464.7 | 0.264 | +10.825 |
| 152012 | slip | loss | 152.4 | 0.388 | -0.049 |
| 152015 | slip | loss | 375.3 | 0.052 | -11.612 |
| 152031 | slip | loss | 44.9 | 0.821 | -3.647 |
| 151005 | tipped | loss | 85.7 | 0.698 | -0.057 |
| 151006 | tipped | loss | 108.3 | 0.121 | -0.027 |

Positive actual cost gain means the selected three-command sequence scored better than the unchanged sequence. Negative gain means it scored worse. This cost combines distance, object height, held-object label, and workspace presence. It is a short-horizon score, not a measure of final task success.

Across the 137 candidate endpoints within the supported workspace, per-axis position RMSE was 76.9 mm. The table also includes objects thrown outside the workspace. Its unbounded errors cannot be compared directly with the workspace-filtered errors in the version 1 report.

The selected sequence had positive measured gain in both old wins and negative gain in all five old losses. This indicates that candidate ranking is a problem in these selected cases. It does not establish a rate across all trials. The held output is a clipped regression score; it is not a calibrated probability.

The repeated baseline branch reproduced its endpoint within 0.000001 m in every case. The audit ran 175 distinct candidate branches plus seven repeat checks. Full actions, predicted positions, held scores, measured endpoints, and branch checks are in prediction_audit.json.

The frozen version 2 test changes only how grasp offsets are executed. It does not repair the model predictions. Its outcome will show whether command consistency helps despite these prediction errors. A later experiment should collect contact and slip examples, calibrate the held score, and test candidate ranking on separate development scenes before another fresh confirmation.

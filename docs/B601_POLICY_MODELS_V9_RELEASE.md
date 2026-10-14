# B601 policy models V9

This archive contains the B601-RS V1–V19 checkpoints, available evaluation records, and selected dataset manifests. V19 has 427,798 parameters. Its checkpoint was selected at epoch 24. Inference measured 4.73 ms per action on the training environment.

V19 weights blue-cube instructions 1.5 times. On 120 fresh matched simulation scenes, V19 placed 150/360 cubes and completed 14/120 full sorts. V18 placed 130/360 cubes and completed 8/120 on the same scenes. The paired placement result is uncertain: 41 wins, 30 losses, 49 ties, p = 0.235. V17 placed 119/360 and completed 7/120; V19's comparison with V17 is promising (52 wins, 31 losses, 37 ties, p = 0.028).

These results do not establish that V19 is better than V18. More fresh evaluations are needed. All policies were tested on the B601-RS PyBullet model. No physical arm was used. The archive does not contain rendered training images; the source repository documents how to recreate the data.

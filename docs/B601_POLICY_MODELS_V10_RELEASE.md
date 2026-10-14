# B601 policy models V10

This archive contains the B601-RS V1–V19 checkpoints, evaluation records, and selected dataset manifests. V19 has 427,798 parameters and weights blue-cube instructions 1.5 times.

V19 was tested on two independent fresh sets of 120 scenes, with V18 on the same seeds. On the first set, V19 placed 150/360 cubes and completed 14/120 full sorts; V18 placed 130/360 and completed 8/120 (41 wins, 30 losses, 49 ties; p = 0.235). On the pre-registered confirmation set, V19 placed 145/360 and completed 9/120 sorts; V18 placed 124/360 and completed 9/120 (49 wins, 34 losses, 37 ties; p = 0.124).

Both sets favor V19 on cube placements, but neither paired test establishes a gain over V18. The archive includes per-seed records for both tests and the confirmation report. All results use the B601-RS PyBullet model. No physical arm was used. The archive does not contain rendered training images; the source repository documents how to recreate them.

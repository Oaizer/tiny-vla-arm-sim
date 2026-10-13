# B601 policy models V8

This bundle contains V1–V18 B601-RS policy checkpoints, available evaluation reports, per-seed results, and selected dataset manifests. The checkpoint uses 427,798 parameters and runs at about 1.89 ms per action on CPU in the training environment.

V18 used 50 intervention-guided DAgger episodes. Nine episodes completed all three placements. On the same 120 held-out simulation seeds, V18 placed 131/360 cubes and completed 6/120 full sorts. V17 placed 125/360 cubes and completed 5/120. The paired cube-placement result is uncertain (41 wins, 35 losses, 44 ties; exact p = 0.567). V18 has the highest observed count so far, but it is not a proven gain over V17.

The bundle does not include rendered training image arrays. The repository documents the scripts needed to recreate them. All results use the B601-RS PyBullet model; no physical arm was used.

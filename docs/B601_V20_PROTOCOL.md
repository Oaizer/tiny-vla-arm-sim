# B601 V20 evaluation protocol

A 12-scene teacher pilot compared TCP grasp heights of 0.08 m, 0.09 m, and 0.10 m. At 0.10 m, the teacher placed 33/36 cubes and completed 9/12 sorts. At the current 0.08 m height, it placed 23/36 and completed 2/12 sorts. The paired seed-level placement test favored 0.10 m (7 wins, 0 losses, 5 ties; exact p = 0.0156). Treat this as pilot evidence; it uses only 12 scenes.

V20 tests the higher 0.10 m pick height. Recollect 100 overhead demonstrations at seeds 620000–620099 and 100 more at seeds 680000–680099. Collect 10 V19-state DAgger episodes at seeds 730000–730009 and 40 more at seeds 730100–730139. All four sets use 384-pixel overhead images and the 0.10 m pick height. Keep face-aligned yaw, model, optimizer, loss weights, and training schedule fixed. Keep V19's blue-cube weight at 1.5. Select the checkpoint with the same validation score as V19.

Evaluate fixed V19 and V20 checkpoints on the same 120 fresh scenes, seeds 661000–670011. Run V19 first and reuse its teacher records for V20. The primary measure is per-trial cube placements with an exact two-sided sign test that excludes ties. Also report full sorts, per-color outcomes, contact presence, wrist-to-face yaw error, cube-local corner proximity, contact height, and joint-limit violations. For corner proximity, 0 means contact at the middle of a cube face and 1 means contact at a corner. Keep task success and contact geometry as separate measures.

All results use the B601-RS PyBullet model. No physical arm is used.

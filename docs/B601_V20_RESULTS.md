# B601 V20 results

## Result

V20 uses a 0.10 m pick height in its teacher data. It keeps V19's model size, camera input, blue-cube loss weight, optimizer, and training schedule. V20 trained on 5,440 samples from 200 training episodes; V19 used 7,152 samples. The model has 427,798 parameters and runs in 4.98 ms per action on CPU.

On 120 fresh, matched B601-RS simulation scenes, V20 placed 131 of 360 cubes and completed 8 of 120 full sorts. V19 placed 165 cubes and completed 16 full sorts on the same scenes. V20 had more placements in 30 trials, fewer in 46, and tied in 44. The exact paired sign test gives p = 0.085. This is a lower observed count for V20, but the test does not show a clear difference at the 0.05 level.

Both policies respected the arm's joint limits. The scripted teacher placed 323 of 360 cubes and completed 88 of 120 sorts on these scenes.

## What the contact audit shows

The V20 policy's mean contact-to-corner score fell from 0.594 to 0.537. A score of 0 is the center of a cube face; 1 is a corner. V20 also had fewer objects above 0.65, from 129 to 98.

Other grasp measures got worse. Mean wrist-to-face yaw error rose from 14.20 to 16.06 degrees. The number of objects with no finger contact rose from 49 to 62, and two-finger contacts fell from 290 to 263. Both runs still had about 94% of measured contacts within 5 mm of the cube's top or bottom rim. The height change moved contacts closer to face centers in the horizontal plane, but it did not remove rim contact.

This points to a mixed result. The V20 policy had a lower measured corner distance, while its yaw alignment and contact rate were worse. Its training set also differed in content and size, so this is not a clean test of grasp height alone. The task result does not support calling V20 an improvement. The contact measurements do not establish a single cause for each failed grasp.

## Next experiment

Keep V19's data and pick height fixed. Add a separate loss weight for wrist yaw on the three grasp waypoints, then compare that model on matched fresh seeds. This tests yaw weighting without changing the data volume or grasp height. Keep the contact audit enabled so the next result can show whether the fingers land nearer the middle of a cube face and whether the policy still closes on both fingers.

All results use PyBullet and the simulated Seeed reBot B601-RS model. No physical arm was used. The 120-scene data summary is in [`portfolio/b601-v20-evaluation.json`](../portfolio/b601-v20-evaluation.json).

The [matched seed 661001 video](../portfolio/media/b601-rs-v19-v20-matched-seed-661001-v19.mp4) shows V19's three placements. The matching [V20 clip](../portfolio/media/b601-rs-v19-v20-matched-seed-661001-v20.mp4) shows one placement.

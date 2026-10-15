# B601 V22 results

## Main result

V22 uses the V19 checkpoint. During approach, descend, close, and lift, it replaces the model's grasp XY and yaw with estimates from the overhead color mask and calibrated camera. It leaves grasp height, gripper width, carry, place, and release under model control. The model has 427,798 parameters and runs in 4.73 ms per action on CPU.

On the first 120 matched simulation scenes, V22 placed 222 of 360 cubes and completed 27 of 120 full sorts. V19 placed 156 cubes and completed 18 full sorts on the same scenes. V22 had more placements in 60 trials, fewer in 26, and tied in 34. The exact paired sign test gives p = 0.000317. Both policies respected all joint limits.

The first test's full-sort result was less certain. V22 had more full sorts in 20 trials, fewer in 11, and tied in 89 (p = 0.150).

An independent confirmation used 120 new matched scenes, seeds 674000–674119. V22 placed 219/360 cubes and completed 27/120 full sorts. V19 placed 141/360 and completed 8/120. V22 had more placements in 67 scenes, fewer in 16, and tied in 37 (p = 1.39 × 10⁻⁸). For full sorts, V22 won 26 scenes, lost 7, and tied in 87 (p = 0.00132). This repeats the placement gain and supports a full-sort gain on the confirmation set.

## Why the grasps looked edge-based

The contact audit supports a clear part of the explanation. On the first test, V19's mean wrist-to-face yaw error was 14.86°. Its mean contact-to-corner score was 0.559, where 0 means the middle of a face and 1 means a corner. V22 reduced these values to 7.04° and 0.466. Objects with a score above 0.65 fell from 93 to 38. Objects with no finger contact fell from 64 to 21, while two-finger contacts rose from 275 to 334.

The confirmation set showed the same pattern: yaw error fell from 16.20° to 7.58°, mean corner proximity fell from 0.590 to 0.470, and objects with no finger contact fell from 80 to 14. Two-finger contacts rose from 255 to 340.

This shows that the learned pose often missed the center of a cube face or turned the gripper away from it. The camera correction improved both measures and more cubes reached their pads. It did not remove every edge-like contact: about 94% of contacts for both policies were within 5 mm of the cube's top or bottom rim. The audit measures contact position; it does not prove why the remaining rim contacts occur.

The first test's [seed 673003 V19 video](../portfolio/media/b601-rs-v19-v22-matched-seed-673003-v19.mp4) shows no placements. The [V22 clip](../portfolio/media/b601-rs-v19-v22-matched-seed-673003-v22.mp4) shows all three. In the independent set, [seed 674007 V19](../portfolio/media/b601-rs-v19-v22-confirmation-seed-674007-v19.mp4) places none; [V22](../portfolio/media/b601-rs-v19-v22-confirmation-seed-674007-v22.mp4) places all three.

## Next step

Keep V22 as the lead simulator policy. The next experiment should improve complete-sort reliability by checking grasp state and recovery before moving to the next object. Keep the learned policy and camera correction separate in future tests so each change has a clear result.

All results use the simulated Seeed reBot B601-RS model in PyBullet. No physical arm was used. The [compact evaluation data](../portfolio/b601-v22-evaluation.json) includes both matched tests, per-seed results, and contact summaries. The full raw contact records stay in local `artifacts/` output.

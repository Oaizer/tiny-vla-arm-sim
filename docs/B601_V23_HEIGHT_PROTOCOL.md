# B601 V23 grasp-height protocol

## Question

Does raising the V22 close waypoint reduce edge contact and improve cube placement?

V22 corrects grasp XY and yaw from overhead pixels. It keeps the learned TCP height. The confirmation contact audit found that the median finger contact was about 19.5 mm above each cube's center, close to the top edge of a 40 mm cube. The TCP itself sat below the cube center, so lowering it was the wrong direction.

## Pilot

Compare V22 with a vertical offset on 12 matched scenes, seeds 675000–675011. Change only the descend and close TCP height. Keep the V19 checkpoint, camera correction, orientation, and delivery actions fixed. Reuse the same teacher records.

Pilot totals were 18/36 placements for V22, 12/36 at −10 mm, 24/36 at +10 mm, and 27/36 at +20 mm. At +20 mm, 5/12 scenes were full sorts, compared with 0/12 for V22. This is a small pilot used to choose one candidate; it does not establish a gain. The placement comparison was 8 wins, 2 losses, and 2 ties (p = 0.109). Contact height and corner proximity did not improve clearly.

## Held-out test

Use fresh matched seeds 676000–676119. Run V22 first and save its teacher records. Then run V23 on the same seeds with a +20 mm offset on only the descend and close waypoints. Do not tune the offset against these scenes.

The primary measure is cubes placed per scene, compared with an exact paired sign test. Also report full sorts, wrist-to-face yaw error, corner proximity, contact height relative to the cube center, finger contact, and joint-limit checks. A lower contact-height magnitude supports more central vertical contact. Report a change as useful only if the task measures improve without a material loss in safe contact or joint-limit behavior.

All runs use the simulated Seeed reBot B601-RS in PyBullet. No physical arm is used. The learned checkpoint stays frozen; V23 tests one control adjustment on top of V22.

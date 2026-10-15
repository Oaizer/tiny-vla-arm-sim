# B601 V22 protocol

V20 and V21 did not improve the learned policy. The V20 contact audit showed more grasp attempts without finger contact and higher yaw error. A 12-scene pilot then used the V19 model with a camera-based grasp correction. It changed only grasp waypoints: the overhead color mask estimates the cube center and face yaw, and the calibrated camera maps the image center to world XY. The pilot placed 23/36 cubes, compared with V19's 15/36. Mean yaw error fell from 15.72° to 8.17°, and objects with no finger contact fell from 6 to 2. This pilot is exploratory; it is not the held-out test.

V22 keeps V19's trained checkpoint and delivery waypoints. On approach, descend, close, and lift, it replaces the model's grasp XY and yaw with the overhead camera estimate. It leaves TCP height, gripper width, and carry, place, and release outputs under model control. The camera uses the saved B601 overhead view and the known projection matrix. No simulator object pose is used by the correction.

Evaluate V19 and V22 on 120 fresh matched scenes, seeds 673000–673119. Run V19 first and reuse its teacher records for V22. Enable the grasp audit on both. The main measure is cubes placed per scene, with an exact paired sign test. Also report full sorts, outcomes by color, wrist-to-face yaw error, corner proximity, finger contact, rim contact, and joint-limit violations. Keep the V19 and V22 clip settings the same.

All trials use PyBullet and the simulated Seeed reBot B601-RS. No physical arm is used.

## Independent confirmation

After the first comparison, a second fresh matched set used seeds 674000–674119. Both checkpoints and the V22 correction stayed fixed. V19 ran first; V22 reused its teacher records. The same contact audit and exact paired sign test were used. No correction tuning used these scenes. See the [confirmation result](B601_V22_RESULTS.md#independent-confirmation).

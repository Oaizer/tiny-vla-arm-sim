# B601-RS simulation and challenge log

The earlier B601 visual proxy has been removed. It solved arm poses from a Panda tool pose each frame, which could jump between inverse-kinematics solutions and did not use B601 physics.

The current pilot loads the Seeed B601-RS URDF directly in PyBullet. The robot’s six joints and two prismatic fingers move in the physics simulation. The scene uses a table, object collision, and the arm’s mesh collision model. The official URDF and meshes are pinned in `src/tinyvla/assets/rebot_b601_rs/`; see its README and included CERN-OHL-W-2.0 licence.

## Reproduce the pilot

From the project root, with the project dependencies installed:

```bash
PYTHONPATH=src python scripts/demo_b601_trials.py
```

To rebuild only the three selected videos, add `--videos-only`.

The script runs 12 seeded cube placements, saves three selected videos under `portfolio/media/`, and writes per-seed results to `artifacts/b601-rs-pilot/` and `portfolio/b601-rs-results.json`. The latter is tracked so the portfolio data is available on GitHub. The scripted teacher reads object and target positions from the simulator. It does not use the VLA or world model.

## Motion and gripper

- The arm uses the URDF’s six joint limits and torque ratings.
- Joint references follow quintic profiles capped at 1.50 rad/s. The cap is on the commanded reference; the simulation does not enforce or validate the real motor speed response. The original V1 pilot used a 2.25 rad/s peak reference and is kept as historical evidence.
- The two gripper fingers use their separate prismatic joints. This pilot commands their URDF ranges directly. It does not reproduce Seeed’s motor-to-width conversion or claim millimetre-accurate gripper width.
- The physics engine resolves finger, object, and table contacts. Arm self-collision is disabled for performance. Seeed documents a web speed range up to 1.50 rad/s and a gripper width range of 0–71.5 mm in its [B601-RS digital twin guide](https://wiki.seeedstudio.com/rebot_arm_b601_rs_web_simulator_developer_guide/).

## Randomized sorting challenge series

The challenge uses three colored cubes and three matching pads on the articulated B601-RS. Each trial randomizes cube positions, cube yaw, and pick order. The task only passes when all three cubes finish within 35 mm of their matching pad. The scripted teacher reads exact simulator positions. It is not a trained policy.

All versions use fresh seed batches. These are small, unpaired samples, so compare them as engineering checks rather than statistical proof. The seed-by-seed records are linked below.

| Version | Change | Complete sorts | Cubes placed | Readout |
| --- | --- | ---: | ---: | --- |
| V2 | Broad randomized starts | 3/5 | 12/15 | Edge starts exposed reach and transfer failures. |
| V3 | Narrower start band | 1/5 | 11/15 | Narrowing alone did not fix the motion. |
| V4 | Lower approach and carry poses | 2/5 | 10/15 | Lower poses reduced IK error but brought the gripper closer to clutter. |
| V5 | Core start band and higher carry | 1/5 | 9/15 | This change alone did not prevent collisions. |
| V6 | Return to the folded joint pose between picks | 9/12 | 31/36 | This produced the best full-sort rate so far. |
| V7 | Keep the V6 return, remove tested edge starts, and keep starts 75 mm from their matching pads | 9/12 | 33/36 | Same full-sort rate as V6; two more cubes placed. Best object-level result so far. |
| V8 | Use a denser, smaller start region | 6/12 | 29/36 | More crowded starts reduced task success. |
| V9 | Add a simulated grasp check before transfer | 9/12 | 31/36 | The check did not improve full-sort success. |

### Barrier-clear transfer V1

The next challenge adds a solid 45 mm wall between the randomized starts and the three matching pads. The same articulated B601-RS model and simulator-state teacher are used. Each of 12 fresh seeds randomizes cube starts, yaw, and pick order. The arm carries at 200 mm and may retry a placement twice.

The result is **8/12 complete sorts and 30/36 cubes placed**. Seed 610000 was used once to check an earlier, taller wall before the final batch was fixed; the other 11 seeds were not used in that check. The open-table V7 run scored 9/12 and 33/36, but it used different seeds and no wall. Treat this as a harder task result, not a paired comparison or a policy improvement. Four trials still had at least one failed placement. The challenge is useful for exposing path clearance and recovery limits.

The selected clip is [seed 610001](../portfolio/media/b601-rs-obstacle-seed-610001-hd.mp4). It uses a 1280 × 720 PyBullet render plus a title strip (1280 × 768 output) and plays at half speed, with denser frames around each grasp. Per-seed starts, object errors, and outcomes are in [the tracked result file](../portfolio/b601-rs-obstacle-v1-results.json).

Reproduce the batch and selected video from the project root:

```bash
PYTHONPATH=src:scripts python scripts/demo_b601_obstacle.py --trials 12 --no-video
PYTHONPATH=src:scripts python scripts/demo_b601_obstacle.py --videos-only --video-seed 610001
```

The wall top is at 145 mm above the simulation origin, while the table top is 100 mm. The teacher uses a 140 mm approach height and a 200 mm carry waypoint. This waypoint is a scripted route choice; the simulation does not guarantee a minimum clearance for every arm link or model the real motor controller.

V6 and V7 complete 9 of 12 trials each. V7 places 33 of 36 cubes, compared with 31 of 36 for V6. The batches use different seed ranges, so this is not a paired comparison. V7 is the current best setting, not a confirmed policy improvement. The arm still fails on some randomized scenes.

### Sort V2 evidence

The first three-cube batch used a broad 4-by-7 start grid. It completed 3 of 5 full sorts and placed 12 of 15 cubes. Failures clustered near workspace edges; retries did not recover off-table cubes.

### Sort V3 evidence

The start grid moved inward, but the old 220 mm approach pose remained. The batch completed 1 of 5 sorts. A forward-kinematics check found 50–88 mm of pose error at that approach height across the tested grid. The grasp-height pose had much lower error.

### Sort V4 evidence

The approach and carry poses both moved down to 140 mm. The IK fit improved, but full-sort success was 2 of 5. Lower carry made it easier for the arm and held object to meet clutter during transfers.

### Sort V5 evidence

This batch capped the start x coordinate at 310 mm and raised carry height to 200 mm. It completed 1 of 5 sorts. A higher carry pose by itself did not prevent the joint-space motion from disturbing other objects.

### Sort V6 evidence

The arm returned to its folded joint pose after each placement. On 12 fresh seeds, it completed 9 full sorts and placed 31 of 36 cubes. This was the highest complete-sort count at that point.

### Sort V7 evidence

V7 kept the folded-pose return, removed the tested far start column, and required each cube to start at least 75 mm from its matching pad. It completed 9 of 12 sorts and placed 33 of 36 cubes. The complete-sort count matches V6; the object count is two higher. Seeds differ, so treat this as a promising engineering result, not a paired gain.

### Sort V8 evidence

V8 tested a denser 18-slot start region. It completed 6 of 12 sorts and placed 29 of 36 cubes. More scene density made the task harder for this motion sequence.

### Sort V9 evidence

V9 added a simulated height and position check after lifting. If the teacher does not detect a held cube, it stops before crossing the table. The batch completed 9 of 12 sorts and placed 31 of 36 cubes. The check did not improve task success over V6 or V7.

Each version has a tracked result file with starts, pick order, object placement error, retries, joint-limit status, and peak reference speed:

- [V2 per-seed data](../portfolio/b601-rs-sort-v2-results.json)
- [V3 per-seed data](../portfolio/b601-rs-sort-v3-results.json)
- [V4 per-seed data](../portfolio/b601-rs-sort-v4-results.json)
- [V5 per-seed data](../portfolio/b601-rs-sort-v5-results.json)
- [V6 per-seed data](../portfolio/b601-rs-sort-v6-results.json)
- [V7 per-seed data](../portfolio/b601-rs-sort-v7-results.json)
- [V8 per-seed data](../portfolio/b601-rs-sort-v8-results.json)
- [V9 per-seed data](../portfolio/b601-rs-sort-v9-results.json)
- [Barrier-clear transfer V1 per-seed data](../portfolio/b601-rs-obstacle-v1-results.json)

### Reproduce the sorting challenges

Run the V7 batch without video capture:

```bash
PYTHONPATH=src python scripts/demo_b601_sort.py --version v7 --trials 12 --no-video
```

Build its selected trial clip:

```bash
PYTHONPATH=src python scripts/demo_b601_sort.py --version v7 --videos-only --video-seed 607000
```

The sorting script supports V2 through V9. Keep each result attached to its version and seed range. V9 checks the simulated object height after a lift and stops before transfer when the grasp check fails. The barrier challenge runs from its own script. All runs use a 1.50 rad/s cap on joint reference speed. This is not a motor speed limit, because the PyBullet position servo does not reproduce Seeed’s online motor controller.

## Current evidence

The one-cube pilot completed 12 of 12 seeded placements. The best open-table three-cube batch completed 9 of 12 full sorts. The added barrier challenge completed 8 of 12. All use a scripted teacher. They do not test a learned VLA policy or real-arm safety. The [tracked per-seed results](../portfolio/b601-rs-results.json) include object starts, placement errors, and joint-limit checks. The older VLA benchmarks are still Panda results and remain listed separately on the portfolio page.

Train a B601-specific policy from this arm’s simulated camera and joint data; the earlier Panda policy does not transfer directly. The demonstration pilot and first learned baseline are described below. All tests in this project remain in simulation. The motor speed profile and gripper width conversion still need calibration before any later hardware integration.

## B601 demonstration data pilot

The recorder saves one sample before each scripted waypoint command. Each sample has an RGB image, six measured arm-joint positions and velocities, two measured finger positions, the current object-to-pad instruction, and the next six arm-joint and two finger-joint targets. These are waypoint labels, not 240 Hz motor commands.

The earlier teacher held the wrist at one fixed yaw while cube yaw changed. That can put the fingers across corners instead of opposing faces. The data collector now rotates the wrist to match each cube's yaw. This uses privileged simulator orientation to choose the teacher action; the student still receives only its image, proprioception, and text. This grasp-alignment change is limited to the data collection script so earlier benchmark results stay reproducible.

The scripted teacher uses exact object and pad poses to choose its targets. Those poses are not included in the saved policy observations. The dataset manifest records teacher success for each whole sorting episode, and assigns every fifth seed to validation. Keep all samples from an episode together when changing this split.

The first 200-episode dataset used 96 × 96 images. It had 7,929 waypoint samples and 144 successful teacher episodes. The first learned controller matched waypoints well, but placed 0/36 cubes in 12 held-out sorts. Cube centers occupied only a few pixels, which made recovery after small movement errors difficult. This is a negative but useful baseline.

The second dataset uses 192 × 192 camera images and a vision encoder that keeps an 8 × 8 spatial feature map. Collect it on a Python 3.11 environment with project dependencies installed:

```bash
PYTHONPATH=src:scripts python scripts/collect_b601_demos.py --episodes 200 --seed-start 630000 --workers 4 --image-size 192 --output artifacts/b601-demos-v2-192
PYTHONPATH=src python scripts/audit_b601_demos.py artifacts/b601-demos-v2-192
PYTHONPATH=src python scripts/train_b601_bc.py --dataset artifacts/b601-demos-v2-192 --output artifacts/b601-bc-v2-192
PYTHONPATH=src:scripts python scripts/evaluate_b601_bc.py --checkpoint artifacts/b601-bc-v2-192/policy.pt --trials 12 --seed-start 631000 --output artifacts/b601-bc-v2-192/heldout-evaluation.json
```

The output goes to `artifacts/b601-demos-v2-192/`, which is ignored by Git because rendered image data can be large. Use a new empty `--output` directory for each collection run. Four workers collect independent seeds in parallel; lower `--workers` if CPU use is too high. The audit checks image and state shapes, finite values, B601 action limits, manifest counts, and episode-level train/validation separation. The 96 × 96 baseline remains in `artifacts/b601-demos-v1/` for comparison.

The behavior-cloning baseline is a small CNN for the camera, token embeddings for the instruction, and an MLP for joint state. It predicts the next six arm-joint and two gripper-joint waypoint targets. Training uses a held-out 20% split kept at the episode level. Validation scores measure waypoint imitation. `evaluate_b601_bc.py` then runs the policy on new seeds and compares it with the scripted teacher. The policy gets eight learned waypoints per object and has no retry logic; the teacher can retry up to three times. Task success is the meaningful closed-loop measure, while the training report also records parameter count, checkpoint size, and CPU inference latency. Test the 96 × 96 and 192 × 192 results as separate versions.

## Learned B601 policy iterations

The same 12 held-out seeds (621000–621011) were used for each policy test. The scripted teacher completed 5 of 12 sorts and placed 26 of 36 cubes. It can retry a pick up to three times. Each learned policy gets eight waypoints per object and no retries.

| Version | Change | Learned sorts | Cubes placed | Teacher sorts |
| --- | --- | ---: | ---: | ---: |
| BC V1 | 96 × 96 images, mixed demonstrations | 0/12 | 0/36 | 5/12 |
| BC V2 | 192 × 192 images; all demonstrations | 0/12 | 0/36 | 5/12 |
| BC V2 success-only | Train on complete teacher episodes | 0/12 | 1/36 | 5/12 |
| BC V3 | Remove return-home labels and add an eight-step phase input | 0/12 | 0/36 | 5/12 |
| BC V4 | Add 960 teacher corrections from policy-visited states | 0/12 | 0/36 | 5/12 |
| BC V5 | 384 px images; 12 × 12 vision grid | 0/12 | 0/36 | 5/12 |
| BC V5b | Smaller 8 × 8 vision grid | 0/12 | 0/36 | 5/12 |
| BC V6 | Predict Cartesian waypoints and face yaw | 0/12 | 0/36 | 5/12 |
| BC V7 | Overhead camera with Cartesian waypoints | 0/12 | 1/36 | 5/12 |
| BC V8 | Add 40 DAgger correction episodes | 0/12 | 0/36 | 5/12 |
| BC V9 | Add image-derived cube and pad targets | 0/12 | 2/36 original; 6/36 fresh | 5/12 |
| BC V10 | Combine image targets with DAgger corrections | 0/12 | 5/36 original; 3/36 fresh | 5/12 |
| BC V11 | Correct camera yaw handedness | 0/120 | 7/36 + 10/36 + 10/36 + 8/36 + 6/36 + 4/36 + 5/36 + 8/36 + 7/36 + 7/36 | 80/120 |
| BC V12 | Add corrections from V11-visited states | 0/12 | 10/36 original; 10/36 + 3/36 fresh | 5/12 |
| BC V13 | Train on successful teacher episodes only | 0/12 | 2/36 original; 2/36 + 0/36 fresh | 5/12 |
| BC V14 | Weight descend, close, and lift errors 3× | 0/36 | 13/108 | 21/36 |
| BC V15 | Weight carry, place, and release errors 3× | 0/36 | 5/108 | 21/36 |
| BC V16 | Add 100 fresh teacher episodes | 1/120 | 9/36 + 10/36 + 15/36 + 8/36 + 13/36 + 11/36 + 9/36 + 8/36 + 13/36 + 13/36 | 80/120 |
| BC V17 | Add teacher corrections from V16 states | 5/120 | 125/360 | 80/120 |

The 192 × 192 model had 425,880 parameters and a 0.58–0.69 ms CPU forward pass. V3 had 426,904 parameters and a 0.74 ms CPU forward pass. V4 stayed at 426,904 parameters and reached 0.69 ms per forward pass. These values measure model inference only, not camera capture or arm movement. Waypoint error did not predict task success, so none of these policies replaces the scripted teacher in the portfolio's best-result summary.

The first teacher used one fixed wrist yaw while cube yaw changed. The fingers could then meet cube corners. The demonstration collector now turns the wrist to align the fingers with each cube's faces. The teacher uses the simulator's cube orientation only to make its labels. The learned policy still receives only the camera image, joint state, language instruction, and waypoint phase.

V3 removes the ninth return-home command from the imitation labels because evaluation performs that reset outside the policy. It adds an explicit phase index for approach, descend, close, lift, carry, place, release, and retreat. V4 adds labels queried from the teacher on states reached by the V3 policy. This is DAgger-style data collection. It did not improve held-out task success. The per-version metrics and seeds are in the [portfolio iteration record](../portfolio/iterations.json).

The 384 × 384 overhead runs use Cartesian position, wrist yaw, and finger width targets. V7 placed one cube on the original 12-seed test. It did not complete a full sort. V8 added 40 DAgger episodes from V7 states. The teacher labeled 960 waypoints, but none of those episodes completed all three placements. V8 placed 0/36 cubes on the original test.

V9 adds eight values computed from the camera image and instruction: the requested cube center, its square orientation, the named pad center, and detection flags. It does not read simulator object poses. On the original seeds, V9 placed 2/36 cubes, compared with 1/36 for V7. On a fresh paired set of 12 seeds, V9 placed 6/36 cubes and V7 placed 0/36. Neither policy completed a full sort in either group. Across 24 matched seeds, that is 8/72 placements for V9 and 1/72 for V7, with 0/24 full sorts for each. This is an early object-level gain. The sample is small, and cubes from the same trial can share failure causes.

V10 combines those image features with the V8 correction data. It placed 5/36 cubes on the original seeds and 3/36 on the fresh group. V9 placed 2/36 and 6/36 on those groups. Both placed 8/72 cubes across 24 paired seeds and completed 0/24 full sorts. V10 did not improve the total placement count.

An orientation audit then found that the overhead camera's vertical axis reverses the image angle. V9 and V10 encoded a +20° world yaw as about −20°. V11 conjugates the image square's fourth moment so the detected face angle matches the world angle modulo 90°. A PyBullet camera test checks five rotations from −35° to +35° and requires less than 3° error. Historical checkpoints retain their original feature convention for fair reruns.

V11 changes only that yaw feature and uses the same 80-episode demonstration set and training schedule as V9. On the original 12 seeds it placed 7/36 cubes, followed by 10/36 and 10/36 on two fresh paired sets. V9 placed 2/36, 6/36, and 6/36 on those same seed groups. Across 36 matched trials, V11 placed 27/108 cubes and V9 placed 14/108; neither completed a full sort. V11 had more placements in 15 trials, fewer in 6, and tied in 15. An exact sign test gives p = 0.078, so this is a promising result that needs more trials. The angle correction improves object placements in these tests, but the learned sorter is not solved.

V12 collected 40 DAgger episodes from V11, with 960 teacher labels and 768 labels used for training. None of the V11 episodes completed a full sort. V12 placed 10/36, 10/36, and 3/36 cubes on the three matched cohorts, for 23/108 total. V11 placed 27/108 on those same seeds. V12 and V11 each placed more cubes on 11 trials, with 14 ties, so the corrections did not produce a task gain.

V13 kept only teacher-successful demonstration episodes: 56 train episodes and 14 validation episodes. Its waypoint loss was lower, but it placed 2/36, 2/36, and 0/36 cubes on the same cohorts. The 4/108 total was far below V11's 27/108. This shows why waypoint loss alone is not a useful checkpoint selection measure for this task.

V14 kept the V9 demonstration set and multiplied waypoint loss in the descend, close, and lift phases by three. It selected checkpoints by this weighted validation loss. On the same 36 seeds, V14 placed 13/108 cubes, compared with V11's 27/108. V14 had more placements in 4 trials, fewer in 13, and tied in 19 (exact paired sign test p = 0.049). Neither policy completed a full sort. Extra emphasis on grasp waypoints did not improve task outcomes, so V11 remains the leading learned policy. The result is in the [portfolio B601 policy data](../portfolio/b601-bc-results.json).

V15 shifted the 3× weight to carry, place, and release. It placed 5/108 cubes on the same 36 seeds, compared with V11's 27/108. V15 had more placements in 3 trials, fewer in 19, and tied in 14 (exact paired sign test p = 0.00086). This weighting made delivery outcomes worse.

V16 tests data volume instead of loss weighting. We collected 100 fresh teacher episodes with new randomized seeds; 75 episodes were complete sorts and the set contained 3,448 waypoint labels. V16 used the 80 training episodes from that set with the original 80 training episodes. The original 20-episode validation split stayed unchanged. The model has 5,424 training samples and the same 427,798 parameters as V11. Across 120 matched held-out seeds, V16 placed 109/360 cubes and completed one full sort. V11 placed 72/360 cubes and completed none. V16 had more placements in 49 trials, fewer in 17, and tied in 54 (exact paired sign test p = 0.00010). This supports the larger dataset for object placements. Full sorting remains rare, so the policy is still an experimental simulator result.

V17 adds 40 DAgger episodes collected from V16 policy states. The teacher labeled 960 waypoints, with 768 entering training. Across 120 matched seeds, V17 placed 125/360 cubes and completed 5/120 full sorts. V16 placed 109/360 cubes and completed 1/120. The paired placement result was 47 wins, 40 losses, and 33 ties (exact sign test p = 0.520). The added corrections do not show a reliable placement gain over V16, though V17 has the best observed count at that point.

V18 changes DAgger collection after the V17 policy completed only 1/40 correction episodes. The first takeover prototype converted teacher joint targets back to Cartesian and solved IK again. Even teacher-only rollouts failed, so that path was discarded. The collector now holds each grasp pose for the full pick-and-place sequence and uses the scripted teacher's Cartesian waypoint directly when it takes over. It records teacher labels on policy-visited states and takes control if the predicted position differs by more than 25 mm, equivalent square yaw differs by more than 0.25 rad, or gripper opening differs by more than 15 mm. A 10-episode pilot and a 40-episode collection produced 9 complete sorts and 1,200 labels; 960 labels from the train split entered training. V18 has 232 training episodes, 7,152 samples, 427,798 parameters, and 1.89 ms CPU inference per action.

On the same 120 held-out seeds, V18 placed 131/360 cubes and completed 6/120 sorts. V17 placed 125/360 cubes and completed 5/120. V18 had more cubes in 41 trials, fewer in 35, and tied in 44 (exact paired sign test p = 0.567). Against V16, V18 placed 131 versus 109 cubes (51 wins, 36 losses, 33 ties; p = 0.133). The observed count is the best so far, but neither comparison proves a gain over the more recent policies. Against V11, V18's gain is clearer: 131 versus 72 cubes (60 wins, 24 losses, 36 ties; p = 0.00011). This remains simulation evidence; full sorting is still uncommon.

The [V18 comparison report](../artifacts/b601-bc-v18-intervention-dagger/COMPARISON.md) includes paired outcomes and exact tests. A dual-camera [V18 full-sort video](../portfolio/media/b601-rs-bc-v18-seed-626006-dual.mp4) shows seed 626006. The collector, report, policy, and 120 per-seed records are available in the project repository and release bundle.

V19 tests a 1.5× loss weight for blue-cube instructions, using the same data, model, optimizer, and training schedule as V18. It has 427,798 parameters. On 120 fresh matched seeds, V19 placed 150/360 cubes and completed 14/120 full sorts. V18 placed 130/360 and completed 8/120; V19 had more placements in 41 trials, fewer in 30, and tied in 49 (exact sign test p = 0.235). The difference from V18 remains uncertain. V19 improved blue placements from 50 to 55 and green placements from 33 to 45, while red placements changed from 47 to 50. V17 placed 119/360 on the same new seeds; V19's paired result was 52 wins, 31 losses, and 37 ties (p = 0.028). This is a promising result against V17, not confirmation that V19 is better than V18. More fresh seeds should be tested before making that claim. The [V19 report](../artifacts/b601-bc-v19-blue-weight/COMPARISON.md), [protocol](B601_V19_PROTOCOL.md), and [dual-camera video](../portfolio/media/b601-rs-bc-v19-seed-632001-dual.mp4) record this experiment.

The pre-registered confirmation test kept both checkpoints fixed and used another 120 fresh matched seeds. V19 placed 145/360 cubes, compared with V18's 124/360; both completed 9/120 sorts. V19 had more placements in 49 trials, fewer in 34, and tied in 37 (p = 0.124). The direction matches the first test, but this result is also inconclusive. By color, V19/V18 placements were red 46/52, blue 57/39, and green 42/33. See the [confirmation protocol](B601_V19_CONFIRMATION_PROTOCOL.md) and [full result](../artifacts/b601-bc-v19-blue-weight/CONFIRMATION.md).

V20 raised the teacher pick height to 0.10 m. On 120 fresh matched seeds, V20 placed 131/360 cubes and completed 8/120 sorts. V19 placed 165/360 and completed 16/120 on the same scenes. V20 had more placements in 30 trials, fewer in 46, and tied in 44 (p = 0.085), so the observed drop is not conclusive. V20's contacts were closer to face centers on average, but its mean wrist-to-face yaw error and count of objects without finger contact were higher. The data set also differed in size and content from V19's, so this was not a clean test of pick height alone. See the [V20 results](B601_V20_RESULTS.md) and [portfolio comparison data](../portfolio/b601-v20-evaluation.json).

V21 multiplied wrist-yaw loss by 3 on descend, close, and lift waypoints while keeping V19's data and schedule fixed. On 120 fresh matched seeds, V21 placed 112/360 cubes and completed 3/120 full sorts; V19 placed 149/360 and completed 7/120. V21 had 31 placement wins, 54 losses, and 35 ties (p = 0.0165). Mean yaw error rose from 15.04° to 16.01°, and objects with no finger contact rose from 55 to 97. This training change did not improve the policy. See the [V21 protocol](B601_V21_PROTOCOL.md) and [portfolio evaluation](../portfolio/b601-v21-evaluation.json).

V22 keeps V19's checkpoint and corrects grasp XY and yaw from the overhead color mask and calibrated camera. On the first 120 matched seeds, V22 placed 222/360 cubes versus V19's 156/360. On an independent 120-seed confirmation, it placed 219/360 versus 141/360. The paired placement results were 60 wins, 26 losses, and 34 ties (p = 0.000317) in the first set, and 67 wins, 16 losses, and 37 ties (p = 1.39 × 10⁻⁸) in the confirmation. Full sorts were 27/120 in both V22 runs; the paired test was uncertain in the first set (p = 0.150) and favored V22 in confirmation (26 wins, 7 losses, 87 ties; p = 0.00132). Mean yaw error fell from 14.86° to 7.04° in the first set and 16.20° to 7.58° in confirmation. See the [V22 results](B601_V22_RESULTS.md) and [protocol](B601_V22_PROTOCOL.md).

V23 adds 20 mm to V22's camera-corrected descend and close waypoints. On 120 fresh matched seeds, it placed 260/360 cubes and completed 47/120 full sorts; V22 placed 226/360 and completed 31/120. V23 had 49 placement wins, 22 losses, and 49 ties (p = 0.00182). Full sorts had 31 wins, 15 losses, and 74 ties (p = 0.0259). Mean absolute contact height fell only slightly, from 18.95 mm to 18.57 mm; most contacts stayed near the cube rims. See the [V23 results](B601_V23_HEIGHT_RESULTS.md) and [protocol](B601_V23_HEIGHT_PROTOCOL.md).

The separate teacher run on seed 607000 shows face-aligned grasps on all three cubes. It is a scripted simulator-state teacher, not a learned-policy result. The V9 clip on seed 622009 and V11 clip on seed 623004 each place two cubes and miss the third. These clips use 1280 × 768 video. The page links to [all seed-level policy records](../portfolio/b601-bc-results.json).

Reproduce V8 through V13 from the project root. DAgger validation remains separate from the original demonstration validation set. V11's yaw convention is checked by the camera test in `tests/test_b601_bc.py`:

```bash
PYTHONPATH=src python scripts/audit_b601_demos.py artifacts/b601-dagger-v8-overhead-40
PYTHONPATH=src python scripts/train_b601_bc.py --dataset artifacts/b601-demos-v7-overhead384 --extra-training-dataset artifacts/b601-dagger-v8-overhead-40 --output artifacts/b601-bc-v8-overhead-dagger --action-mode cartesian --spatial-size 8 --epochs 80 --seed 20261013
PYTHONPATH=src:scripts python scripts/evaluate_b601_bc.py --checkpoint artifacts/b601-bc-v8-overhead-dagger/policy.pt --trials 12 --seed-start 621000 --teacher-results artifacts/b601-bc-v7-overhead-cartesian/heldout-evaluation.json --output artifacts/b601-bc-v8-overhead-dagger/heldout-evaluation.json
PYTHONPATH=src python scripts/train_b601_bc.py --dataset artifacts/b601-demos-v7-overhead384 --output artifacts/b601-bc-v9-overhead-visual-targets --action-mode cartesian --spatial-size 8 --epochs 40 --seed 20261014 --visual-target-features
PYTHONPATH=src python scripts/train_b601_bc.py --dataset artifacts/b601-demos-v7-overhead384 --extra-training-dataset artifacts/b601-dagger-v8-overhead-40 --output artifacts/b601-bc-v10-overhead-visual-dagger --action-mode cartesian --spatial-size 8 --epochs 50 --seed 20261015 --visual-target-features
PYTHONPATH=src python scripts/train_b601_bc.py --dataset artifacts/b601-demos-v7-overhead384 --output artifacts/b601-bc-v11-overhead-visual-yaw --action-mode cartesian --spatial-size 8 --epochs 40 --seed 20261016 --visual-target-features
PYTHONPATH=src:scripts python scripts/evaluate_b601_bc.py --checkpoint artifacts/b601-bc-v11-overhead-visual-yaw/policy.pt --trials 12 --seed-start 621000 --teacher-results artifacts/b601-bc-v7-overhead-cartesian/heldout-evaluation.json --output artifacts/b601-bc-v11-overhead-visual-yaw/heldout-evaluation.json
PYTHONPATH=src:scripts python scripts/evaluate_b601_bc.py --checkpoint artifacts/b601-bc-v11-overhead-visual-yaw/policy.pt --trials 12 --seed-start 622000 --teacher-results artifacts/b601-bc-v7-overhead-cartesian/heldout-replication-622000.json --output artifacts/b601-bc-v11-overhead-visual-yaw/heldout-replication-622000.json
PYTHONPATH=src:scripts python scripts/evaluate_b601_bc.py --checkpoint artifacts/b601-bc-v11-overhead-visual-yaw/policy.pt --trials 12 --seed-start 623000 --teacher-results artifacts/b601-bc-v9-overhead-visual-targets/heldout-replication-623000.json --output artifacts/b601-bc-v11-overhead-visual-yaw/heldout-replication-623000.json
PYTHONPATH=src:scripts python scripts/collect_b601_dagger.py --checkpoint artifacts/b601-bc-v11-overhead-visual-yaw/policy.pt --episodes 40 --seed-start 660000 --output artifacts/b601-dagger-v11-yaw-40
PYTHONPATH=src python scripts/audit_b601_demos.py artifacts/b601-dagger-v11-yaw-40
PYTHONPATH=src python scripts/train_b601_bc.py --dataset artifacts/b601-demos-v7-overhead384 --extra-training-dataset artifacts/b601-dagger-v11-yaw-40 --output artifacts/b601-bc-v12-overhead-visual-dagger --action-mode cartesian --spatial-size 8 --epochs 50 --seed 20261017 --visual-target-features
PYTHONPATH=src python scripts/train_b601_bc.py --dataset artifacts/b601-demos-v7-overhead384 --output artifacts/b601-bc-v13-overhead-visual-success-only --action-mode cartesian --spatial-size 8 --epochs 40 --seed 20261018 --visual-target-features --successful-only
PYTHONPATH=src python scripts/train_b601_bc.py --dataset artifacts/b601-demos-v7-overhead384 --output artifacts/b601-bc-v14-overhead-visual-grasp-weighted --action-mode cartesian --spatial-size 8 --epochs 40 --seed 20261019 --visual-target-features --grasp-phase-weight 3.0
PYTHONPATH=src python scripts/train_b601_bc.py --dataset artifacts/b601-demos-v7-overhead384 --output artifacts/b601-bc-v15-overhead-visual-delivery-weighted --action-mode cartesian --spatial-size 8 --epochs 40 --seed 20261020 --visual-target-features --delivery-phase-weight 3.0
PYTHONPATH=src:scripts python scripts/collect_b601_demos.py --episodes 100 --seed-start 680000 --workers 4 --image-size 384 --camera-view overhead --output artifacts/b601-demos-v15-overhead384-extra
PYTHONPATH=src python scripts/audit_b601_demos.py artifacts/b601-demos-v15-overhead384-extra
PYTHONPATH=src python scripts/train_b601_bc.py --dataset artifacts/b601-demos-v7-overhead384 --extra-training-dataset artifacts/b601-demos-v15-overhead384-extra --output artifacts/b601-bc-v16-overhead-visual-more-demos --action-mode cartesian --spatial-size 8 --epochs 40 --seed 20261021 --visual-target-features
PYTHONPATH=src python scripts/collect_b601_dagger.py --checkpoint artifacts/b601-bc-v16-overhead-visual-more-demos/policy.pt --episodes 40 --seed-start 700000 --output artifacts/b601-dagger-v16-expanded-40
PYTHONPATH=src python scripts/train_b601_bc.py --dataset artifacts/b601-demos-v7-overhead384 --extra-training-dataset artifacts/b601-demos-v15-overhead384-extra --extra-training-dataset artifacts/b601-dagger-v16-expanded-40 --output artifacts/b601-bc-v17-overhead-visual-more-demos-dagger --action-mode cartesian --spatial-size 8 --epochs 40 --seed 20261022 --visual-target-features
```

Recreate the intervention-guided V18 training data and checkpoint with:

```bash
PYTHONPATH=src:scripts python scripts/collect_b601_dagger.py --checkpoint artifacts/b601-bc-v17-overhead-visual-more-demos-dagger/policy.pt --episodes 10 --seed-start 710020 --output artifacts/b601-dagger-v17-intervention-fixed-pilot-10
PYTHONPATH=src:scripts python scripts/collect_b601_dagger.py --checkpoint artifacts/b601-bc-v17-overhead-visual-more-demos-dagger/policy.pt --episodes 40 --seed-start 710100 --output artifacts/b601-dagger-v17-intervention-fixed-40
PYTHONPATH=src python scripts/train_b601_bc.py --dataset artifacts/b601-demos-v7-overhead384 --extra-training-dataset artifacts/b601-demos-v15-overhead384-extra --extra-training-dataset artifacts/b601-dagger-v16-expanded-40 --extra-training-dataset artifacts/b601-dagger-v17-intervention-fixed-pilot-10 --extra-training-dataset artifacts/b601-dagger-v17-intervention-fixed-40 --output artifacts/b601-bc-v18-intervention-dagger --action-mode cartesian --spatial-size 8 --epochs 30 --visual-target-features
PYTHONPATH=src python scripts/train_b601_bc.py --dataset artifacts/b601-demos-v7-overhead384 --extra-training-dataset artifacts/b601-demos-v15-overhead384-extra --extra-training-dataset artifacts/b601-dagger-v16-expanded-40 --extra-training-dataset artifacts/b601-dagger-v17-intervention-fixed-pilot-10 --extra-training-dataset artifacts/b601-dagger-v17-intervention-fixed-40 --output artifacts/b601-bc-v19-blue-weight --action-mode cartesian --spatial-size 8 --epochs 30 --visual-target-features --blue-cube-weight 1.5
```

Evaluate the checkpoint on the same ten 12-seed blocks used for V17, reusing the saved teacher results. The first block uses `heldout-evaluation.json`; blocks 622000 through 630000 use `heldout-replication-<seed>.json`. Save V18 outputs to `artifacts/b601-bc-v18-intervention-dagger/heldout-replication-<seed>.json`, then run `PYTHONPATH=src:scripts python scripts/report_b601_v18.py` to rebuild the paired report and portfolio data.

The exporter rebuilds the compact portfolio data from the ignored local checkpoints and trial records:

```bash
python scripts/export_b601_bc_results.py
python scripts/update_b601_portfolio.py
```

### 384-pixel experiment

V5 uses 384 × 384 images and a 12 × 12 feature grid. It has 100 demonstration episodes with an episode-level validation split. Collect and train it with:

```bash
PYTHONPATH=src:scripts python scripts/collect_b601_demos.py --episodes 100 --seed-start 650000 --workers 4 --image-size 384 --output artifacts/b601-demos-v5-384
PYTHONPATH=src python scripts/audit_b601_demos.py artifacts/b601-demos-v5-384
PYTHONPATH=src python scripts/train_b601_bc.py --dataset artifacts/b601-demos-v5-384 --output artifacts/b601-bc-v5-384 --epochs 40 --batch-size 32 --spatial-size 12
PYTHONPATH=src:scripts python scripts/evaluate_b601_bc.py --checkpoint artifacts/b601-bc-v5-384/policy.pt --trials 12 --seed-start 621000 --teacher-results artifacts/b601-bc-v1-all/heldout-evaluation.json --output artifacts/b601-bc-v5-384/heldout-evaluation.json
```

The dataset and model files stay under ignored `artifacts/` so Git history does not contain large generated arrays. The portfolio tracks their sample counts, checkpoints, and held-out results in compact JSON files.

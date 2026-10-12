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

The selected clip is [seed 610001](../portfolio/media/b601-rs-obstacle-seed-610001.mp4). It is 384 × 432 and plays at half speed, with denser frames around each grasp. Per-seed starts, object errors, and outcomes are in [the tracked result file](../portfolio/b601-rs-obstacle-v1-results.json).

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

Before policy training, calibrate target speed and gripper width conversion against Seeed’s own B601-RS simulation controller. Then collect B601-RS episodes with joint-state observations and train a fresh policy on this arm. Evaluate it on new seeds against a B601-specific feedback controller. Keep all tests in simulation.

# B601-RS simulation pilot

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
- The position servo moves joint targets along quintic profiles. Segment duration uses 1.2 rad/s, but the quintic profile peaks at up to 2.25 rad/s. Seeed lists 1.50 rad/s as the top web target speed, so this is a simulation pilot, not a faithful copy of its online controller.
- The two gripper fingers use their separate prismatic joints. This pilot commands their URDF ranges directly. It does not reproduce Seeed’s motor-to-width conversion or claim millimetre-accurate gripper width.
- The physics engine resolves finger, object, and table contacts. Arm self-collision is disabled for performance. Seeed documents a web speed range up to 1.50 rad/s and a gripper width range of 0–71.5 mm in its [B601-RS digital twin guide](https://wiki.seeedstudio.com/rebot_arm_b601_rs_web_simulator_developer_guide/).

## Current evidence

The pilot completed 12 of 12 seeded placements in its narrow test area. It shows that the model and trial path work for a scripted teacher. It does not test varied tasks, robustness, learned perception, or the VLA checkpoint. The [tracked per-seed results](../portfolio/b601-rs-results.json) include object starts, placement errors, and joint-limit checks. The older VLA benchmarks are still Panda results and remain listed separately on the portfolio page.

Before policy training, calibrate target speed and gripper width conversion against Seeed’s own B601-RS simulation controller. Then collect B601-RS episodes with joint-state observations and train a fresh policy on this arm. Evaluate it on new seeds against a B601-specific feedback controller. Keep all tests in simulation.

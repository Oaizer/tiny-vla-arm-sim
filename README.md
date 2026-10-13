# TinyVLA for a simulated robot arm

The [simulation portfolio](portfolio/index.html) compares the project iterations and includes selected trial videos.

This project trains a small vision and language policy in PyBullet. The VLA benchmarks use the Franka Panda. A new simulation pilot loads Seeed’s B601-RS URDF as the physical arm and runs scripted pick-and-place trials. The B601-RS has six arm joints and two prismatic gripper fingers. The pilot does not evaluate a learned policy. It runs on a CPU in Linux or Ubuntu under WSL. The second version adds shape recognition, three target pads, placing and lifting tasks, varied scenes, and a feedback controller.

The project also has a small world model trained only on simulated arm episodes. It predicts future observations and compares nearby grasp commands. See the [world model guide](docs/WORLD_MODEL.md), [work plan](docs/WORLD_MODEL_PLAN.md), and [world model results](artifacts/world-v1/RESULTS.md).

The [B601-RS simulation work](docs/B601_SIMULATION.md) loads the articulated Seeed arm model, records scripted-teacher demonstrations, and trains a B601-specific behavior-cloning baseline. Run `PYTHONPATH=src python scripts/demo_b601_trials.py` to reproduce the original pilot. The older VLA checkpoints and benchmarks still use Panda. The B601 speed profile and finger-to-width conversion need calibration before any future hardware integration.

The [second world-model experiment](artifacts/world-v2/RESULTS.md) tests a consistent grasp offset. In 720 fresh simulation trials, feedback completed 150 of 200 tasks. Both JEPA variants completed 145 of 200. Each controller stopped before motion in 39 of 40 missing-object trials. Feedback remains the default. Read the [experiment plan](docs/WORLD_MODEL_V2.md) and [prediction audit](artifacts/world-v2/PREDICTION_AUDIT.md). Run `python -m tinyvla.world_iteration run` to reproduce the comparison with the saved model weights.

Nine matched clips from three version 2 trials are in the [video bundle](https://github.com/Oaizer/tiny-vla-arm-sim/releases/download/world-model-v2/tiny-vla-impact-videos.zip). The bundle includes a playback gallery and SHA-256 checksums. The clips show a successful forced-slip recovery, a matched slip trial where both planners fail, and a trial after the object is moved. They are selected examples; the full trial results are in the report.

Version 3 trains a 6,468 parameter contact head from PyBullet finger-force labels. It improves held-out contact prediction, but its top grasp contacts less often than the unchanged command in a 40-scene ranking audit. The head stays offline. Read the [version 3 results](artifacts/world-v3/contact/RESULTS.md) and [experiment log](docs/WORLD_MODEL_LOG.md). Download all derived data and weights from the [version 3 release](https://github.com/Oaizer/tiny-vla-arm-sim/releases/tag/world-model-v3-contact). The bundle includes 480 data episodes, including the preserved 40-scene pilot, plus the 1000-branch corrected audit, model, reports, and hashes. Run `python scripts/verify_world_contact.py` to check the files and recreate the manifest.

## Results

The trained model has **17,896 parameters**. Its checkpoint is **78,815 bytes**. We tested two controllers in 800 trials across ten test sets. The feedback controller succeeded in 39 of 40 standard trials, 32 of 40 trials where an object moved, and 24 of 40 trials with a forced drop. It stopped before moving in 39 of 40 trials where the requested object was missing. Feedback did worse than one-shot control in some changed scenes.

Read the [full results](artifacts/RESULTS.md). Watch the [moved-object demo](artifacts/demo.mp4) or the [slip recovery demo](artifacts/slip-demo.mp4). The feedback rules are hand-written. The model learns to read images and instructions and to predict arm waypoints.

## Run the project

You need Linux or Ubuntu under WSL, Python **3.11.16**, and [uv](https://docs.astral.sh/uv/). The lock file pins each package version and hash. It uses the CPU build of PyTorch.

```bash
bash run.sh
```

This command checks the project, creates a new run folder, generates training and validation data, trains the model, and tests it. It also saves a demo, videos of failed trials, the results, and a file of hashes. It keeps earlier runs. You do not need a GPU or a paid service.

To test an existing model or make a demo:

```bash
uv run --locked tinyvla evaluate --run-dir artifacts --suites nominal object_moved slip
uv run --locked tinyvla demo --run-dir artifacts \
  --instruction 'put the red cylinder on the cyan pad' --event object_moved
```

The included experiment and its files are in `artifacts/`. New runs go to `artifacts/runs/<timestamp>/`.

## Project files

- `src/tinyvla/` contains the language rules, simulator, model, training code, controller, tests, and command line tool.
- `configs/project.json` sets the model size, data, training, and test limits.
- `tests/` checks the supported language, repeatable simulator resets, and safe recovery rules.
- `docs/MODEL_GOAL.md` describes the model and its limits.
- `docs/BENCHMARK.md` describes the scene changes and how we score each trial.
- `baseline-v1/` contains the first version and its report.
- `artifacts/` contains the trained model, data, logs, videos, and results.

## What the model can do

It can identify a red cube, red cylinder, blue cube, or green cylinder. It can place or lift one of these objects. Place commands can name a yellow, purple, or cyan pad. The camera calibration is supplied to the model.

The model only understands the words and sentence forms listed in [the model guide](docs/MODEL_GOAL.md). It does not understand arbitrary language or unfamiliar objects. The camera view is blocked during a grasp, so the controller looks again between steps. It does not track objects through the blockage. We have only tested this model in simulation.

The model predicts image locations, object presence, which object and task the words describe, and arm waypoints. Standard inverse kinematics and joint controls move the arm. The hand-written feedback controller looks at a new camera image between steps. It checks the finger sensors after a grasp, retries a limited number of times, and stops when its attempt limit is reached. It cannot read object labels or the simulator's scoring data.

## Change or test the code

```bash
uv sync --locked
uv run --locked python -m unittest discover -s tests -v
```

The GitHub Actions workflow runs the project checks. The Git repository includes the VLA data, model weights, and experiment reports. Download the larger world-model trajectories from the [version 1 release](https://github.com/Oaizer/tiny-vla-arm-sim/releases/tag/world-model-v1). The repository includes their hashes and collection code. The PyBullet arm and other packages keep their own licenses.

Version 4 adds targeted red-cylinder data. On 40 fresh red-cylinder scenes, its contact head reached 0.777 average precision, compared with 0.732 for the version 3 head on the same scenes. In an independent 40-scene ranking check, its top grasp made contact in 20 scenes, compared with 22 for the unchanged command. The head ranked candidates slightly better than the old score, but did not beat the unchanged command. It remains offline. Read the [version 4 results](artifacts/world-v4/contact/RESULTS.md) and [experiment plan](docs/WORLD_MODEL_V4_PLAN.md). The [version 4 release](https://github.com/Oaizer/tiny-vla-arm-sim/releases/tag/world-model-v4-contact) includes both ranking cohorts, its targeted data, and the version 3 base data needed to reproduce training.

Version 5 trains a 241-parameter grasp reranker. On two fresh 40-scene checks it chose contact in 56/80 scenes, compared with 51/80 for the unchanged grasp. The result is a small contact-ranking lead, not a task-success gain. The reranker stays offline. Read the [version 5 results](artifacts/world-v5/reranker/RESULTS.md), [experiment plan](docs/WORLD_CONTACT_RANKER_PLAN.md), and [replication plan](docs/WORLD_CONTACT_RANKER_REPLICATION_PLAN.md). The [portfolio page](portfolio/index.html) brings the iterations and selected trial videos together. Download the [version 5 data and weights](https://github.com/Oaizer/tiny-vla-arm-sim/releases/download/world-model-v5-reranker/tiny-vla-world-v5-reranker.zip) or read the [release notes](https://github.com/Oaizer/tiny-vla-arm-sim/releases/tag/world-model-v5-reranker).

Version 6 adds an experimental hybrid controller. Feedback acts first. The world model can propose a small red-cylinder grasp change during recovery. The feedback controller checks the result and keeps its bounded retry and stop rules. On 40 matched simulation trials, feedback and hybrid each completed 26 tasks, with all 40 outcomes tied. The hybrid stays offline. An audit found a contact-head channel mismatch and a weak tipped-object input. Read the [V6 results](artifacts/world-v6/hybrid-feedback/RESULTS.md) and [frozen protocol](artifacts/world-v6/hybrid-feedback/protocol.json) before using it.

Versions 7–12 tested corrected contact features, policy fine-tuning, recovery fallback, visual policy routing, and different planner gates. None improved task success. Version 13 changed the training label: it ranked grasps by full simulated pick, lift, and place success. Across two fresh matched tests, feedback completed 51/80 tasks and V13 completed 57/80. The paired result was 11 wins, 5 losses, and 64 ties (exact p=0.210). This is a repeatable positive result in one simulation task, but the sample is small. The feedback controller remains the default. Read the [V13 report](artifacts/world-v13/task-ranker/REPLICATION.md), [offline ranking results](artifacts/world-v13/task-ranker/OFFLINE_RESULTS.md), and [frozen protocols](artifacts/world-v13/task-ranker/), and download the [V13 model, data, and videos](https://github.com/Oaizer/tiny-vla-arm-sim/releases/download/world-model-v13-task-success/tiny-vla-world-v13-task-success.zip). Run `uv run --locked python scripts/verify_world_hybrid_v13.py` to check the source hashes and paired counts.

The [simulation portfolio](portfolio/index.html) now tracks all 23 iterations, including failed approaches and selected videos. Panda VLA experiments and the B601-RS physics pilot use simulation. No physical robot was used.

Version 14 removes the initial approach-and-retreat probe. On 40 matched hybrid trials, the original and direct-approach controllers both completed 21 tasks, with 40 paired ties. The new variant used 33 fewer policy calls, but did not reduce control time. The [V14 report](artifacts/world-v14/direct-approach/RESULTS.md) and matched video pair are in the portfolio. The selected target arm is Seeed’s reBot B601-RS. The articulated-arm simulation pilot and new B601 behavior-cloning baseline are recorded in the [B601-RS report](docs/B601_SIMULATION.md) and [portfolio](portfolio/index.html). The older learned VLA tests still use Panda. Hardware speed and gripper calibration are future work.

The first B601-RS pilot completed 12/12 single-cube placements. Later randomized sorting tests reached 9/12 complete three-cube tasks, with 33/36 objects placed in the best batch. Those challenge scores use a simulator-state teacher. The new learned B601 policy is measured separately in the [B601-RS simulation log](docs/B601_SIMULATION.md#b601-demonstration-data-pilot). Joint reference speed is capped in simulation, but motor speed and gripper-width conversion are not calibrated.

V11 corrects the overhead image-yaw direction. It placed 27/108 cubes across the original 36 held-out seeds. V12's V11-state corrections placed 23/108; V13's successful-only training placed 4/108. V14 weighted grasp errors three times and placed 13/108; V15 weighted delivery errors three times and placed 5/108. Across 120 matched seeds, V16 placed 109/360 cubes, compared with V11's 72/360 (49 wins, 17 losses, 54 ties; p = 0.00010). V17 added corrections from V16 states and placed 125/360 cubes. It completed 5/120 full sorts; V16 completed 1/120. V17's lead over V16 is still uncertain (47 wins, 40 losses, 33 ties; p = 0.520). The [portfolio](portfolio/index.html) shows the results and dual-camera trial videos. All results are from simulation.

Download the [B601 policy checkpoints and evaluation bundle](https://github.com/Oaizer/tiny-vla-arm-sim/releases/download/b601-policy-models-v7/b601-policy-models-v7.zip). It contains model weights, per-seed results, and dataset manifests. It does not include large rendered training image arrays; use the documented scripts to recreate them.

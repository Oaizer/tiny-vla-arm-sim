# TinyVLA v2 results

The CPU model has **17,896 learned parameters**. Its checkpoint is **78,623 bytes**.
Training took 0.5 seconds. The horizontal action error on validation data was 123.30 mm RMSE.

The feedback controller checks a new image before it grasps and before it places an object. It uses the finger sensors to detect an empty grasp, retries a limited number of times, and then stops. These recovery rules are hand-written. The model learns to read images and commands and to predict arm waypoints.

| Test set | One-shot successes | Feedback successes | Feedback retries |
|---|---:|---:|---:|
| Standard scenes | 0/2 | 0/2 | 0 |

The slowest p95 model inference time across the test sets was 6.28 ms on this CPU. This does not include rendering, arm calculations, or movement.

Each pair of results uses the same scene seeds and commands. After each trial, the scorer checks the simulator state. The model and controller cannot read this scoring data. In the moved-object test, the scene moves the object once after the arm approaches. In the slip test, the scene opens and closes the gripper once after the lift.

The JSON summaries show results by task and object, confidence intervals, model timing, and checkpoint hashes. The JSONL files keep each trial, command, retry, stop reason, and controller event. Videos show the first trials listed in the configuration.

## Scope

The task has four object types (red cube, red cylinder, blue cube, and green cylinder), three pad colors, and two task types. The command grammar and camera calibration are provided. The model does not receive object positions. Test sets change scene features that were also varied during training. This small model does not support open-ended language. We tested it only in simulation.

The v1 result of 94 successes in 100 trials came from easier scenes with different objects, pads, and camera settings. It cannot be compared directly with these v2 results.

## Reproducibility

See `manifest.json` for source and artifact hashes, the software environment, Git revision, settings, and seed ranges. Use the lock file and `tinyvla run` to create a new experiment. The Git repository includes this run and its data.

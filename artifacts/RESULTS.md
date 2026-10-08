# TinyVLA v2 experiment results

Trained CPU model: **17,896 parameters**, **78,815 checkpoint bytes**.
Training time: 472.4 seconds; validation horizontal action RMSE: 2.07 mm.

The feedback supervisor replans from fresh RGB before grasping and before placement, detects empty grasps using finger encoders, and retries within a fixed attempt budget. The recovery logic is scripted; perception, instruction selection, and waypoint heads are learned.

| Suite | One-shot success | Feedback success | Feedback retries |
|---|---:|---:|---:|
| nominal | 40/40 | 39/40 | 1 |
| lighting | 33/40 | 34/40 | 1 |
| camera | 40/40 | 38/40 | 1 |
| appearance | 33/40 | 33/40 | 8 |
| dynamics | 38/40 | 37/40 | 1 |
| positions | 38/40 | 33/40 | 2 |
| language | 37/40 | 36/40 | 1 |
| object_moved | 0/40 | 32/40 | 14 |
| slip | 0/40 | 24/40 | 44 |
| missing source (safe stop) | 39/40 | 39/40 | 0 |

Slip recovery by object: red cube: 11/12, red cylinder: 1/12, blue cube: 8/8, green cylinder: 4/8. Fallen cylinders remain a substantial failure case.

Maximum per-suite p95 neural inference latency: 5.73 ms on this CPU. This excludes rendering, inverse kinematics, and robot motion.

Each pair uses matching scene seeds and instructions. Success is measured by simulator state only after execution; that state is unavailable to the policy/supervisor. Missing-source rows measure stopping before any task motion. The moved-object event relocates the object once after approach; the slip event opens and recloses the gripper once after lifting.

Detailed JSON summaries include task/object breakdowns, Wilson confidence intervals, inference latency, and checkpoint hashes. JSONL files retain all trials, commands, supervisor events, retries, and stop reasons. Videos show the first configured trials per suite.

## Scope

Four object identities (red cube, red cylinder, blue cube, green cylinder), three pad destinations, two task families, and a declared command grammar. Camera calibration is supplied as an input; object coordinates are not. Images and dynamics vary within training ranges; individual benchmark suites apply held-out shifts. This is a compact structured visuomotor system, not open-vocabulary language or a foundation VLA. No physical arm was used.

The original 94/100 result describes a different, easier v1 scene distribution and is not directly comparable to these expanded v2 suites.

## Reproducibility

See `manifest.json` for the exact source/artifact hashes, environment, Git revision, configuration, and seed ranges. Use the repository lockfile and `tinyvla run` to reproduce a fresh experiment. Generated artifacts are excluded from Git but retained beside this report.

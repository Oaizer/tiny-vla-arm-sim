# TinyVLA v1 results

We trained and tested this model on the local CPU in Ubuntu under WSL. No cloud training or paid compute was used. The table reports one test run.

| Measure | Result |
|---|---:|
| Held-out trials | 100 |
| Successful pick and place | **94 / 100** |
| Success rate | **94%** |
| 95% Wilson interval | 87.5%–97.2% |
| Learned parameters | **16,104** |
| Checkpoint size | **69,687 bytes** |
| CPU inference, median | **1.51 ms** |
| CPU inference, p95 | **2.63 ms** |
| Training updates | 3,000 |
| Training time | 208.3 seconds |
| Training scenes / labeled action chunks | 1,200 / 7,200 |
| Validation scenes | 120 |
| Horizontal validation error after fitting the action head | 1.53 mm RMSE |

Inference time covers tensor preparation and one model prediction. It does not include image capture, rendering, physics, or arm movement. It can change with CPU load.

## Results by command

| Command | Successful trials |
|---|---:|
| Red cube to yellow pad | 15 / 17 |
| Red cube to purple pad | 16 / 17 |
| Green cube to yellow pad | 16 / 17 |
| Green cube to purple pad | 15 / 17 |
| Blue cube to yellow pad | 16 / 16 |
| Blue cube to purple pad | 16 / 16 |

## Input checks

We ran the same 30 extra scene seeds in each row. We did not use these trials to change the model.

| Test | Success against the original goal |
|---|---:|
| Correct image and command | 30 / 30 |
| Command names a different cube | 0 / 30 |
| Camera image is blank | 0 / 30 |

The results show that this policy used the camera image and color words in these tests. They do not show that it understands general language or can handle unfamiliar objects and cameras. When a command names a different cube, success is scored against the original goal, so failure is expected.

## Files and failed trials

- [Demo video](demo.mp4): a new scene, seed 30000, red cube to yellow pad.
- [Failed trial video](evaluation/failure_20093.mp4): green cube to purple pad; the grasp failed.
- `evaluation/episodes.jsonl`: all 100 trial records.
- `evaluation/summary.json`: summary results.
- `runs/tiny/train.jsonl`: training log.
- `runs/tiny/refit.json`: action-head fitting results.
- `environment.lock.txt`: package versions.
- `runs/tiny/best.pt`: model checkpoint.

The first six test videos show one trial for each command. The six failed seeds were 20033, 20037, 20062, 20072, 20084, and 20093. Each failed to lift the intended cube above 10 cm. A replay video can differ slightly because object contact depends on the physics step.

## What this result covers

This was a small simulation test with a fixed overhead camera, upright colored cubes, two pads, six commands, and one action plan per image. The arm is the Panda model bundled with PyBullet. Simulated finger contact holds objects; no code attaches them to the gripper. Standard inverse kinematics and joint controls carry out the learned waypoints.

The training set contains scripted action targets for generated scenes. We did not validate all 7,200 targets by running them in simulation. We ran the teacher in 12 simulation episodes before generating the data.

This is a small model trained from scratch. It is not a fine-tuned SmolVLA model or a LIBERO result. We did not test it on a physical arm. It does not adjust its path during movement, avoid obstacles, understand open-ended commands, or guarantee success when scenes change.

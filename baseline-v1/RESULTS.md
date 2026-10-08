# Completed simulation experiment

The model was trained and evaluated locally on the user's machine using the CPU under Ubuntu WSL. No cloud training or paid compute was used. Source code, training data, logs, checkpoints, and videos are included.

| Measurement | Result |
|---|---:|
| Held-out physics trials | 100 |
| Successful pick and place | **94 / 100** |
| Success rate | **94%** |
| 95% Wilson interval | 87.5%–97.2% |
| Trainable parameters | **16,104** |
| Saved checkpoint size | **69,687 bytes** |
| CPU policy inference, median | **1.51 ms** |
| CPU policy inference, p95 | **2.63 ms** |
| Neural training updates | 3,000 |
| Neural training elapsed time | 208.3 seconds |
| Training scenes / labelled action chunks | 1,200 / 7,200 |
| Validation scenes | 120 |
| Horizontal validation error after head refit | 1.53 mm RMSE |

Latency measures tensor conversion plus a policy forward pass, not camera rendering, physics stepping, or the seconds needed to complete a movement. It was measured during this local run and may vary with CPU load.

## Task breakdown

| Instruction | Successes / trials |
|---|---:|
| Red block → yellow pad | 15 / 17 |
| Red block → purple pad | 16 / 17 |
| Green block → yellow pad | 16 / 17 |
| Green block → purple pad | 15 / 17 |
| Blue block → yellow pad | 16 / 16 |
| Blue block → purple pad | 16 / 16 |

## Checks that the inputs matter

The same 30 extra scene seeds (21000–21029) were run in each condition. These trials were not used to change the model.

| Condition | Success against the original goal |
|---|---:|
| Correct image and instruction | 30 / 30 |
| Instruction requests a different block | 0 / 30 |
| Camera image replaced by zeros | 0 / 30 |

These checks support that this policy depends on both vision and the color words. They do not establish general language understanding or robustness to unfamiliar cameras or objects. Changing the instruction intentionally changes the requested behavior; failure against the original goal is the expected outcome.

## Evidence

- [Annotated learned-policy demo](demo.mp4): new scene seed 30000, red block to yellow pad, successful lift and placement.
- [Recorded failure](evaluation/failure_20093.mp4): green block to purple pad, failed grasp.
- [All 100 trial records](evaluation/episodes.jsonl).
- [Machine-readable results](evaluation/summary.json).
- [Training log](runs/tiny/train.jsonl).
- [Linear head refit results](runs/tiny/refit.json).
- [Exact environment packages](environment.lock.txt).
- [Model checkpoint](runs/tiny/best.pt).

The first six benchmark rollouts were all recorded, one for each instruction, under `evaluation/`. The six failed benchmark seeds were 20033, 20037, 20062, 20072, 20084, and 20093. All six failed to lift the intended block above 10 cm. The failure video replays one of them; small differences in contact dynamics between a standalone replay and an ongoing batch can change exact distances.

## Scope of the result

This completes a deliberately narrow simulation prototype: fixed overhead camera, upright colored cubes, two target pads, six command combinations, and one-shot eight-waypoint actions. The arm is the bundled Panda in PyBullet. The object is held by simulated gripper contact, not a scripted attachment. Standard IK and position servos execute learned Cartesian/gripper commands.

The training set contains scripted action targets for generated scenes, not 7,200 individually rollout-validated demonstrations. The scripted teacher was checked on 12 physical simulation rollouts before data generation. The learned policy was then independently evaluated as documented above.

This is a custom miniature VLA, not a fine-tuned SmolVLA or a LIBERO reproduction. Those were the initial suggested tools; the available machine has an Intel Arc GPU and no NVIDIA CUDA GPU, so this completed run uses a smaller CPU-trainable architecture. No real-world arm transfer has been tested. There is no mid-motion visual correction, obstacle avoidance, open-vocabulary language, or distribution-shift guarantee.

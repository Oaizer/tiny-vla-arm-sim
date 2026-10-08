# Tiny VLA: simulated Panda pick and place

A locally trained, 16,104-parameter vision-language-action policy for a narrow tabletop task. Runs on CPU in Linux or Windows through WSL. It is a custom model trained from scratch, not a SmolVLA checkpoint or a LIBERO benchmark result.

**Measured result:** 94/100 successful unseen pick-and-place trials (95% Wilson interval 87.5–97.2%), 69,687-byte checkpoint, and 1.51 ms median / 2.63 ms p95 CPU inference. See [RESULTS.md](RESULTS.md), [demo.mp4](demo.mp4), and [the checkpoint](runs/tiny/best.pt).

## What it does

Input: one 96 × 96 RGB overhead image, a tokenized instruction, and the Panda gripper pose/opening. Output: eight Cartesian waypoints and finger-opening commands. A conventional inverse-kinematics/position controller executes those commands in PyBullet. The physics engine supplies gravity, collisions, and finger friction; there is no object attachment constraint or teleport during execution.

Supported instruction template: `put the COLOR block on the PAD pad`, with COLOR = red/green/blue and PAD = yellow/purple. Three blocks are present in every scene, so the instruction must select an object and destination. Object positions, pad positions, and block brightness vary between scenes.

## Model

Three convolutional layers encode the image. Learned word embeddings condition two spatial attention maps. Their soft coordinates and the robot state feed a learned linear action head. The complete network emits the waypoint/gripper sequence. Simulator object coordinates are available to the training teacher and outcome scorer only; the policy receives RGB, tokens, and proprioception.

The camera is fixed, blocks are identical upright cubes, and commands follow a small grammar. This is an open-loop action-chunk policy: it plans from the initial observation and does not correct for objects moving during execution. It is not a general language model, a collision planner, or a physical-arm deployment package.

## Reproduce

Requires Linux (or Ubuntu under WSL), `uv`, and roughly 1 GB for the CPU runtime. No GPU, paid services, or cloud credentials are needed.

```bash
bash setup.sh
bash run_pipeline.sh
```

The virtual environment is `/var/tmp/tiny-vla-env`; override `VLA_ENV` to choose another path. On this Windows machine, the project is at `/mnt/c/Users/oaize/Documents/Codex/2026-10-07/i-w/outputs/tiny-vla` inside WSL.

Run just the trained policy evaluation:

```bash
/var/tmp/tiny-vla-env/bin/python evaluate.py \
  --checkpoint runs/tiny/best.pt \
  --episodes 100 --start-seed 20000 --output evaluation
```

Run a specific instruction and produce a video:

```bash
/var/tmp/tiny-vla-env/bin/python demo.py \
  --instruction 'put the blue block on the purple pad' \
  --seed 30001 --output my-demo.mp4
```

## Training and evaluation design

- Train: 1,200 initial scenes, seeds 0–1199, and all six commands per scene (7,200 labelled action chunks).
- Validation: 120 different scenes, seeds 10000–10119; select checkpoint using action prediction error.
- Physics testing: 100 new scenes, seeds 20000–20099, with all six instructions distributed across the run.
- Supervision: scripted teacher uses simulator state to construct action chunks. A sample of teacher rollouts is executed to verify its grasping behavior; generation does not execute every training chunk.
- Loss: action imitation plus auxiliary spatial attention supervision. The auxiliary labels are unavailable during inference.
- After neural training, `refit_head.py` fits the linear action head by ridge regression on the 7,200 training pairs and accepts it only if validation error improves. This reduced horizontal validation RMSE from 5.03 to 1.53 mm. Test scenes were not used for training or checkpoint selection.
- Success: instructed block was lifted above 10 cm, then finished resting within 4.5 cm of the instructed pad center. Logs record final distance and observed lift height. Other cubes are distractors; their displacement is not part of the success metric.
- Ablations: remove image content or change the source-color instruction while keeping the scored goal fixed, to check that the learned policy uses vision and language.

Check `evaluation/summary.json` for measured task success and latency, `evaluation/episodes.jsonl` for every trial, and the MP4 files for recorded rollouts. Training curves are in `runs/tiny/train.jsonl`; the selected model is `runs/tiny/best.pt`.

## Files

- `simulator.py`: Panda environment, teacher, physics controller, scorer.
- `generate.py`: deterministic image/instruction/action data generation.
- `model.py`: complete trainable VLA architecture.
- `train.py`: CPU training and checkpoint selection.
- `refit_head.py`: training-only least-squares fit of the learned action head.
- `evaluate.py`: held-out physics evaluation, ablations, and videos.
- `demo.py`: run one supported instruction and save an annotated video.
- `setup.sh`, `run_pipeline.sh`: reproduce the experiment.

## Next steps toward a physical arm

1. Replace the Panda URDF and controller with the intended arm's geometry and control semantics.
2. Match camera calibration, gripper dimensions, joint limits, and action rates.
3. Add textures, lighting, pose, friction, mass, and camera randomization and test the resulting distribution shifts.
4. Collect real demonstrations and fine-tune; this simulation result alone does not establish real-world transfer.
5. Replace the one-shot chunk with repeated observations and short-horizon replanning before attempting dynamic tasks.

For broader language tasks and more varied scenes, the larger path discussed in the chat is LeRobot + SmolVLA + LIBERO on Linux with suitable training compute. That experiment has not been represented as completed by this project.

## Upstream dependencies

- [PyBullet and the bundled Franka Panda model](https://github.com/bulletphysics/bullet3)
- [PyTorch](https://pytorch.org/)
- [LeRobot LIBERO guide](https://github.com/huggingface/lerobot/blob/main/docs/source/libero.mdx)

Robot assets are supplied by the installed `pybullet_data` package and retain their upstream licensing.

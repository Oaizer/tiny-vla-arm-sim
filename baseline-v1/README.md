# TinyVLA v1: simulated pick and place

This first version trained a 16,104-parameter policy to move a simulated Panda arm. It runs on a CPU in Linux or Ubuntu under WSL. The model was trained from scratch. It is not a SmolVLA checkpoint or a LIBERO benchmark result.

In 100 held-out trials, the model placed the requested cube on the requested pad in 94 trials. The checkpoint is 69,687 bytes. CPU inference took 1.51 ms at the median and 2.63 ms at p95. See the [full results](RESULTS.md), [demo video](demo.mp4), and [checkpoint](runs/tiny/best.pt).

## Task and model

The model reads one 96 × 96 RGB image from an overhead camera, a short command, and the gripper pose and opening. It predicts eight arm and gripper waypoints. PyBullet supplies gravity, collisions, and finger contact. The simulation does not attach or move the object by code during a task.

Commands name a red, green, or blue cube and a yellow or purple pad. Three cubes are present in each scene. Object positions, pad positions, and cube brightness change across scenes.

Three image layers find the target. Learned word embeddings guide the model to the object and pad. A learned action head predicts the waypoints. Object coordinates are used to create training targets and score trials. The model only sees the image, command, and gripper data.

This first version uses a fixed camera, upright cubes, and a small command grammar. It plans from the first image and does not correct its path if an object moves. It is not a general language model, obstacle planner, or tested physical-arm controller.

## Run it

You need Linux or Ubuntu under WSL, `uv`, and about 1 GB for the CPU software.

```bash
bash setup.sh
bash run_pipeline.sh
```

The virtual environment is in `/var/tmp/tiny-vla-env`. Set `VLA_ENV` to use another folder. To test the saved model:

```bash
/var/tmp/tiny-vla-env/bin/python evaluate.py \
  --checkpoint runs/tiny/best.pt \
  --episodes 100 --start-seed 20000 --output evaluation
```

To run one command and save a video:

```bash
/var/tmp/tiny-vla-env/bin/python demo.py \
  --instruction 'put the blue block on the purple pad' \
  --seed 30001 --output my-demo.mp4
```

## Training and tests

- Training used 1,200 scenes (seeds 0–1199) and all six commands in each scene: 7,200 labeled action chunks.
- Validation used 120 other scenes (seeds 10000–10119) to select the model.
- Testing used 100 new scenes (seeds 20000–20099). Each of the six commands appears in the test set.
- A script created action targets from simulated scene state. We ran the scripted teacher in 12 simulations to check its grasp behavior; we did not execute every training action chunk.
- Training combined action imitation with image-location supervision. The extra image labels are not available at run time.
- After training, a linear action head was fit using training data only. Horizontal validation error fell from 5.03 mm to 1.53 mm RMSE. Test scenes were not used to train or select the model.
- A place succeeds if the named cube rises above 10 cm and comes to rest within 4.5 cm of the named pad center.
- Extra checks removed the image or changed the command's object while scoring against the original goal. These checks test whether the model uses the image and color words; they do not show general language skill.

The evaluation folder has trial logs and videos. The training log and checkpoint are in `runs/tiny/`.

## Files

- `simulator.py`: Panda arm, physics, scripted teacher, and scorer.
- `generate.py`: creates repeatable image, command, and action data.
- `model.py`: model layers.
- `train.py`: CPU training and model selection.
- `refit_head.py`: fits the action head using training data.
- `evaluate.py`: held-out tests and videos.
- `demo.py`: runs one supported command and saves a video.
- `setup.sh`, `run_pipeline.sh`: set up and run the experiment.

## Before trying a physical arm

Use the intended arm model and controls. Match the camera, gripper, joint limits, and control rate. Add scene changes and collect real demonstrations. Test new observations and short action steps before trying tasks where objects can move. This simulation result does not show transfer to a real arm.

For broader language and more varied scenes, a separate project could compare LeRobot, SmolVLA, and LIBERO on Linux with suitable training compute. That work is not part of this version.

## Upstream packages

- [PyBullet and the bundled Franka Panda model](https://github.com/bulletphysics/bullet3)
- [PyTorch](https://pytorch.org/)
- [LeRobot LIBERO guide](https://github.com/huggingface/lerobot/blob/main/docs/source/libero.mdx)

The Panda assets come from the installed `pybullet_data` package and retain their upstream license.

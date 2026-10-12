# Seeed reBot Arm B601-RS model

This folder contains Seeed Studio’s B601-RS URDF and meshes from [`Seeed-Projects/reBot-DevArm`](https://github.com/Seeed-Projects/reBot-DevArm), pinned at commit `69de7380656a86440aead27692dc6deee1a7f024`, under `Rebot_Arm_description/RS/`. The included licence is CERN-OHL-W-2.0.

The PyBullet pilot loads this model as the physical arm. It uses the six revolute arm joints, both prismatic finger joints, joint limits, inertia, and mesh collisions. The simulator adds the table and test objects. Joint targets use a quintic profile with a quintic time profile. See the pilot report for its tested speed and the difference from Seeed’s control limits.

The pilot teacher reads exact object positions from simulation. This is useful for checking arm mechanics, grasp contact, trial timing, and video capture. It does not evaluate a learned policy. Existing VLA checkpoints and results use the Panda model and do not transfer to this arm.

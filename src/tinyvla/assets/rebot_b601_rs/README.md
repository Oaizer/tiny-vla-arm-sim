# Seeed reBot Arm B601-RS model

This folder contains Seeed Studio's official B601-RS URDF and mesh files. The source is [`Seeed-Projects/reBot-DevArm`](https://github.com/Seeed-Projects/reBot-DevArm), pinned at commit `69de7380656a86440aead27692dc6deee1a7f024`, under `Rebot_Arm_description/RS/`. The included licence is CERN-OHL-W-2.0.

The model is used as a visual proxy in the current PyBullet scenes. Its six arm joints solve for the pose of the hidden Panda tool. Its gripper opening follows the Panda finger opening. Proxy collisions are disabled. The Panda still handles contact, inverse kinematics, and task scoring. This gives the videos the B601-RS appearance without claiming B601-RS dynamics or benchmark results.

The proxy changes the camera image seen by the current policy. Existing checkpoints were trained with the Panda visible, so they must be retrained and reevaluated before their results can be interpreted with this appearance. Existing saved videos remain unchanged until new trials are recorded.

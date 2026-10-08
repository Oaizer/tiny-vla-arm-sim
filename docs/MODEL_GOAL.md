# Model goal and architecture decision

Build a CPU-deployable learned policy for a declared tabletop vocabulary. Budget: fewer than 100,000 trainable parameters, checkpoint below 1 MB, and target p95 inference below 15 ms on this machine. The physical-arm deployment step is outside this version.

The available machine has an Intel Arc GPU and no NVIDIA CUDA GPU. Training runs on CPU under Ubuntu WSL. A small structured policy lets this project generate, train, and compare models locally without paid services or remote credentials.

## Supported language and tasks

Objects: **red cube**, **red cylinder**, **blue cube**, **green cylinder**. Both red shapes coexist, so color alone does not specify the object.

Destinations: **yellow**, **purple**, and **cyan** pads. Task families: **place** an object on a pad, or **lift** and hold it. There are 16 semantic task combinations.

Accepted grammar:

- `[put|place|move] [the] COLOR SHAPE [on|onto|to] [the] DESTINATION [pad|target]`
- `[lift|raise|pick up] [the] COLOR SHAPE`

The object/color pairing must be one of the four listed above. Training uses 48 phrasings. The language benchmark uses new combinations of known words, including omitted articles and changed verb/preposition pairings. Unknown words and unsupported objects are rejected explicitly. This is a fixed vocabulary, not unrestricted natural-language understanding.

## Learned components

1. A small convolutional network predicts spatial maps and presence for all four objects and three pads from RGB.
2. Word embeddings and a GRU encode the raw instruction. Learned selection logits identify the source, destination, and task family; the grammar parser does not supply those identities to the policy.
3. Spatial attention gives image locations. Known camera calibration projects those locations into world coordinates at the declared object/pad heights.
4. Two learned linear heads predict eight waypoint/finger commands for placing and lifting. After gradient training, the absolute-waypoint heads are refitted on training examples by ridge regression. Pose coefficients are constrained to zero because these teacher targets are absolute target positions, independent of the initial robot pose.

The controller receives proprioception separately for actual pose and finger-opening feedback. Simulator object coordinates are used only for training labels, deliberate benchmark perturbations, and outcome scoring.

## Scripted components

Inverse kinematics and joint position control execute Cartesian commands. A phase supervisor requests new observations, checks learned presence, detects an empty gripper using encoders, retries, and stops at an attempt/inference budget. Recovery rules are designed rather than learned from failure trajectories. Recording failures now provides data for a later learned recovery model.

The overhead view is occluded during a grasp, so the supervisor moves the arm aside before replanning. It replans at phase boundaries rather than continuously tracking arbitrary objects through occlusion. A wrist camera or temporal observation model would be needed for that extension.

## Future alternative

If the project goal changes to open-vocabulary commands or unfamiliar object categories, adopt a pretrained VLA such as SmolVLA, provision suitable training compute, and gather demonstrations for the new embodiment. Compare it against this compact policy using the same physical task metrics. That would be a new training experiment, not a feature claimed by this version.

# Model goal and design

Train a small policy that runs on a CPU and follows a fixed set of tabletop commands. The limits are fewer than 100,000 learned values, a checkpoint below 1 MB, and p95 model inference under 15 ms on this machine. This version is for simulation. It has not been set up for a physical arm.

This computer has an Intel Arc GPU, but no NVIDIA CUDA GPU. Training therefore runs on the CPU in Ubuntu under WSL. This small model can be trained here without cloud services or paid compute.

## Commands the model supports

The objects are a **red cube**, **red cylinder**, **blue cube**, and **green cylinder**. The two red objects appear together, so the model must tell them apart by shape as well as color.

The target pads are **yellow**, **purple**, and **cyan**. The model can place an object on a pad or lift an object and hold it. There are 16 object, pad, and task combinations.

Accepted command forms:

- `[put|place|move] [the] COLOR SHAPE [on|onto|to] [the] DESTINATION [pad|target]`
- `[lift|raise|pick up] [the] COLOR SHAPE`

Training uses 48 different wordings. The language test uses new combinations of the same known words, such as leaving out “the” or changing the verb. The parser rejects words and objects outside this list. This is a small fixed vocabulary, not general language understanding.

## What the model learns

1. A small image network finds each object and pad and estimates whether it is present.
2. Word embeddings and a GRU read the instruction. The model learns which object, pad, and task the words name. The parser checks that a command is allowed; it does not tell the model which object or pad to select.
3. The model finds image locations. A supplied camera calibration maps these locations to positions on the table.
4. Two learned heads predict eight arm and gripper waypoints for a place or lift action. After neural training, we refit these heads using training data only. The target waypoints are absolute, so the fit does not use the starting arm pose.

The controller also reads the arm pose and finger opening. The simulator's object positions are used to create training labels, make test changes, and score the result. The policy and feedback controller do not read those positions.

## What the controller does

Standard inverse kinematics and joint controls move the arm. A hand-written controller asks for new images between action phases, checks whether the object is visible, checks the finger sensors after a grasp, retries, and stops after a fixed number of attempts or model calls. The recovery rules are not learned from failure examples.

The overhead camera cannot see the object clearly while the arm is grasping it. The controller moves the empty arm to a clear view before a grasp, then checks the target pad from the lifted position. It does not track objects through the blocked view. A wrist camera or a model that uses a sequence of images could support that later.

## If the project needs broader language

For commands with unfamiliar words or new object types, try a pretrained VLA such as SmolVLA. That would need more training compute and examples from the intended robot. Test it against this model on the same tasks. This project has not done that training.

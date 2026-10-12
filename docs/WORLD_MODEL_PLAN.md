# World model work plan

## Goal

Train a small model from simulation data. Use it to predict the result of arm commands and choose better grasp moves. Compare it with the existing feedback controller. Current tests use a Panda arm in PyBullet. The next robot target is the Seeed reBot Arm B601-RS. Success means a working, measured experiment; any gain in task success must come from the recorded tests.

## Work order

1. Record 200 pilot episodes, then extend the data to 1,000 episodes. Use 800 for training, 100 for validation, and 100 for prediction tests. Save images, sensor readings, commands, command times, and training labels. Include poor grasps, fallen cylinders, drops, and retries.
2. Keep the current visual network as a fixed feature extractor. Train a small model on three recent observations. First fit it to a fixed target encoder. Then train an online encoder against a slowly updated target encoder. Predict future features and add training labels to preserve useful object information.
3. Predict several future steps. Compare each model with a no-change prediction and with shuffled actions. Check held-out episodes. Select settings using validation and development scenes only.
4. Add a planner to the existing feedback controller. At grasp moves, compare short action sequences. Execute the first command, then observe again. Penalize uncertain predictions and large changes from the existing command.
5. Freeze the selected model and planner. Compare the existing controller and the planner on matching new scene seeds. Save each trial, timing, stops, retries, and videos. Report losses as well as gains.
6. Write the method, results, limits, and run commands in plain English. Push the source, model, reports, and small demo files to the private GitHub repository. Keep large trajectory files in the local output package, with hashes and a command to recreate them.

## Design choices

The existing arm API uses servo commands with different durations. Each training transition will cover one completed servo command. The action includes its duration. This keeps the original controller and physics settings available for a fair comparison. The model does not assume that each step takes the same time.

The camera supplies RGB. Arm position and finger opening are sensor inputs. Known camera calibration is also an input. Simulator object poses are training labels and scoring data only. The deployed planner cannot read them.

The first model uses an action-conditioned JEPA loss: it predicts an encoded future observation. A target encoder changes slowly. Feature variance and auxiliary object and sensor prediction help prevent all images from receiving the same encoding. These extra labels make this a supervised and self-supervised hybrid, not a label-free reproduction of a published model.

The initial planner changes grasp commands only. The existing controller supplies task order, carrying moves, retry limits, and stops. The world model evaluates nearby grasp choices over a short horizon. It cannot predict a sudden external event before that event becomes observable.

## Checks and evidence

- Check that episode seeds do not overlap between splits.
- Check that the action stored for each transition matches the command that ran.
- Test prediction at one and three steps and report errors in useful units.
- Compare the fixed-encoder model, the JEPA model, no-change predictions, and shuffled actions.
- Confirm that the planner accepts only camera, calibration, and sensor inputs.
- Report full planning time as well as model size.
- Use fresh final test seeds. Keep final failures and do not use them to tune the model.

## Known limits

The gripper keeps the original fixed orientation. The camera may lose sight of an object during a grasp. Three observations give limited memory. Good feature predictions do not prove that a plan will succeed. This work must report actual arm outcomes separately.

## Target arm: Seeed reBot B601-RS

A first B601-RS physics pilot now loads Seeed’s URDF directly in PyBullet and tests joint-limited pick-and-place trials. It completed 12 of 12 scripted placements on a narrow randomized set. See the [pilot report](../artifacts/b601-rs-pilot/RESULTS.md) and [simulation notes](B601_SIMULATION.md). The existing learned policy and world model still use Panda observations and actions.

Keep all training and evaluation in simulation. The next step is to define a B601 observation and action format, collect diverse training episodes from this arm, retrain the policy and world model, then evaluate on fresh seeds against a B601-specific feedback baseline. Do not send Panda joint targets or the current checkpoint to the B601-RS.

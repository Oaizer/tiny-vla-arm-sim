# Test plan

Training scenes use seeds 0–2399. Validation scenes use seeds 10000–10199. Each scene provides three sampled commands and image labels for visible objects and pads. Validation selects the saved model. Test scenes use separate groups of seeds that start at 30000.

## Scene changes

| Feature | Training scenes | Shifted test scenes |
|---|---|---|
| Camera | x/y move by up to 4.5 cm; height by up to 7 cm; field of view 40–45° | x/y move by up to 12 cm; field of view 35–48° |
| Brightness | image level 0.75–1.05 | image level 0.40–0.65 |
| Object size and angle | 1.7–2.3 cm; angle ±0.06 rad | 1.5–2.5 cm; angle ±0.30 rad; more pixel noise |
| Object mass | 35–65 g | 80–130 g |
| Surface grip | 0.9–1.3 | 0.55–0.8 |
| Starting area | x 0.36–0.70 m; y −0.24–0.07 m | x 0.34–0.73 m; y −0.27–0.12 m |
| Commands | 48 wordings | New combinations of known words |
| Floor pattern | Random gray 16×16 texture | A new random texture for each scene |

The camera calibration is provided in every test, including tests with changed camera views. These results do not show how the model handles an unknown or incorrect calibration. Floor textures are randomized in both training and testing. We did not test a new family of textures. The simple renderer changes light direction, light level, and pixel noise. It does not add cast shadows.

## How the tests work

Each test set runs two ways to control the arm: **one-shot** and **feedback**. Both use the same saved model, scene seeds, and commands. The one-shot controller carries out its action plan. The feedback controller takes new images between action phases. Each method gets 40 trials in each test set, as set in the project configuration.

Both use the same arm motion. The controller splits each move into steps no longer than 2.5 cm and uses at least 24 physics ticks per step. The direct motion used to reset the scene and make training data is unchanged.

The test sets are:

- `nominal`: new scenes within the training ranges.
- `lighting`, `camera`, `appearance`, `dynamics`, `positions`: one changed scene feature at a time.
- `language`: new sentence combinations made from supported words.
- `object_moved`: after the arm approaches, the test moves the requested object about 7 cm in x and −7 cm in y.
- `slip`: after the arm lifts the object, the test opens and closes the gripper once to make the object drop. The controller is not told that a slip occurred and receives no object position.
- `missing`: the requested object is absent. We check whether the arm stops before it moves toward the task.

A place succeeds if the object rises above 10 cm and then settles within 4.5 cm of the named pad center at a height consistent with resting on it. A lift succeeds if the object stays above 14 cm near the gripper and the fingers are not fully closed. Only the scorer can read the simulator's object positions.

The controller stops if the object presence score is below 0.65. It treats a finger opening below 8 mm as an empty grasp. It allows at most three attempts. These limits are set before testing. Reports include failures and stops, the number of trials for each task and object, and Wilson confidence intervals. Forty trials give only a rough estimate of performance.

The first version's 94% result came from easier tests with three cubes, two pads, and a fixed camera. It is a different experiment. Use the second version's one-shot and feedback results to compare recovery behavior.

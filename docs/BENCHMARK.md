# Benchmark protocol

Train scene seeds: 0–2399. Validation scene seeds: 10000–10199. Every scene contributes three sampled instruction/action labels and visual supervision for all visible objects/pads. Validation selects the checkpoint. Test suites begin at 30000 and use separate 1,000-seed blocks.

## Variation and shifts

| Feature | Training distribution | Held-out suite |
|---|---|---|
| Camera | Eye x/y jitter ±4.5 cm; height ±7 cm; FOV 40–45° | x/y jitter ±12 cm; FOV 35–48° |
| Pixel brightness | multiplier 0.75–1.05 | 0.40–0.65 |
| Object appearance | half-width/radius 1.7–2.3 cm; yaw ±0.06 rad | 1.5–2.5 cm; yaw ±0.30 rad; more pixel noise |
| Mass | 35–65 g | 80–130 g |
| Surface friction | 0.9–1.3 | 0.55–0.8 |
| Object starting area | x 0.36–0.70 m; y −0.24–0.07 m | x 0.34–0.73 m; y −0.27–0.12 m |
| Instruction | 48 templates | new combinations of known words |
| Texture | randomized grayscale 16×16 texture | independently randomized per scene |

Camera calibration is supplied in every condition, including the camera shift. Results do not establish tolerance to unknown or incorrectly calibrated cameras. Texture changes are sampled inside both train and test distributions; there is no separate unseen-texture-family claim. Lighting uses direction/intensity variation and RGB noise, with a deliberately simple renderer and no cast shadows.

## Controller comparisons

Each suite runs **one-shot** execution and **feedback** execution on identical scene seeds and instructions with the same frozen checkpoint. A simulator reset clears physical state and motor history before each episode. The benchmark configuration sets 40 trials per mode per suite.

Both controllers use the same carrying motion controller: Cartesian targets are interpolated in segments no longer than 2.5 cm, with at least 24 physics ticks per segment. Reset/data generation retains its original direct servo motion, so this controller change does not alter the generated training observations.

- `nominal`: fresh scenes inside training ranges.
- `lighting`, `camera`, `appearance`, `dynamics`, `positions`: one held-out shift at a time.
- `language`: held-out sentence combinations inside the declared vocabulary.
- `object_moved`: the harness relocates the requested object by about 7 cm in x and −7 cm in y once after approach. This is an explicit synthetic movement event.
- `slip`: the harness opens and recloses the gripper once after lift to induce a drop. It does not give the controller an object pose or a slip label.
- `missing`: the requested object is absent; measure stopping before any task-motion command.

Placing succeeds if the requested object was raised above 10 cm and then settles within 4.5 cm of the requested pad center, at a height compatible with resting on the plane. Lifting succeeds if that object remains above 14 cm near the gripper with a nonempty finger opening. Privileged object state is accessed only by the outcome scorer.

The source-presence threshold is 0.65, empty-gripper threshold is 8 mm per finger, and maximum recovery attempts is three. These settings are configuration values declared before test evaluation. Reports retain failures and stops rather than discarding episodes. Each task/object breakdown includes its trial count, and summaries include Wilson confidence intervals. Small suite counts provide limited statistical precision.

The v1 prototype's 94% success belongs to its original three-cube/two-pad fixed-camera distribution. Compare one-shot and feedback v2 rows to assess recovery; v1 and v2 percentages are different experiments.

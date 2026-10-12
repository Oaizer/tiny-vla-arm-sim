# B601-RS visual model in simulation

The project includes a separate preview mode that renders Seeed Studio's official reBot B601-RS mesh model. The mesh is packaged under `src/tinyvla/assets/rebot_b601_rs/RS/` with the URDF-relative file layout preserved. See that folder's README and licence for source and reuse terms.

## What the model does

The B601-RS model is a visual proxy in the preview mode. The simulator solves its six arm joints to place the gripper at the hidden Panda tool pose. Its gripper follows the Panda finger opening. The proxy has no collisions. The Panda still drives the scene physics, contact checks, and task score. Run `python scripts/demo_b601_visual.py` to recreate the scripted visual demonstration in the portfolio.

This keeps the change limited to appearance and leaves the existing benchmark simulator unchanged. It does not make old results a B601-RS evaluation. The preview uses the B601-RS image, while the benchmark still uses the Panda image. Old trial videos remain as originally recorded.

## Check the model files

Run this from the repository root:

```bash
python -m tinyvla.b601_model_check
```

The check confirms that the URDF is valid XML, every referenced mesh is present, and the expected six arm joints and two gripper joints are defined. Loading the full scene also needs the project dependencies from the lock file.

## Next step for a B601-RS experiment

Use an RS-specific physics scene, joint limits, actuator settings, and gripper contact model. Then collect fresh training and test episodes and retrain the policy and world model. Do not treat the visual proxy as a dynamics-accurate digital twin.

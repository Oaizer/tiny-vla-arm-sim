# Version 4: focus on red-cylinder contact

## Aim

Version 3 predicted contact better overall, but it missed many red-cylinder contacts. Its top-ranked grasp also contacted less often than the unchanged command. This experiment adds targeted red-cylinder examples and checks candidate ranking again in fresh PyBullet scenes.

The JEPA model and arm controller stay fixed. The contact head stays offline. No physical robot is used.

## Data

Collect 103 training, 24 validation, and 40 test episodes. Every new episode uses the red cylinder and the same yellow pad instruction. The seed ranges are 185000–185102, 185600–185623, and 186200–186239. Use an 80% tipped-object rate, a 45% forced-slip rate, and stronger approach noise than the version 3 collection. Existing version 3 train and validation episodes add variety. Keep the 40 new red-cylinder test episodes separate as the main held-out result. The run uses fewer scenes than first planned because image and sensor collection took longer than expected.

Collect 20 more red-cylinder scenes for ranking with seeds 187000–187019. Test 25 approach and closure branches from each saved state. Compare the learned score with the old geometric score and the unchanged command. Record the actual finger force after closure. These branches are offline simulator copies. They do not control the task policy. Treat this smaller ranking sample as preliminary.

To check the small first audit, keep the trained weights fixed and run a second, independent ranking cohort of 40 red-cylinder scenes with seeds 187020–187059. Do not pool the first cohort into the primary replication result. Save its protocol and data in a separate folder.

## Measures and decision

Report average precision, Brier score, precision, recall, and calibration for the targeted test episodes. Also report results on the combined held-out set, with the per-object rows. For ranking, report top-choice contact rate and candidate average precision. Preserve every episode, failed collection, protocol, model hash, and seed list.

Keep the head offline if the fresh red-cylinder test or ranking result is weak or mixed. A new controller comparison needs a clear gain on fresh ranking scenes first. Do not use the 20 ranking scenes to tune the model.

## Run

From the repository root, use the locked project environment:

```bash
uv run --locked python -m tinyvla.world_contact_targeted collect --split train
uv run --locked python -m tinyvla.world_contact_targeted collect --split validation
uv run --locked python -m tinyvla.world_contact_targeted collect --split test
uv run --locked python -m tinyvla.world_contact_targeted train
uv run --locked python -m tinyvla.world_contact_targeted rank
```

The project environment may need to be activated first. Outputs are written to `artifacts/world-v4/contact/`. The report and ranking protocol record the source and data hashes.

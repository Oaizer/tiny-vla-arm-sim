# Version 4: focus on red-cylinder contact

## Aim

Version 3 predicted contact better overall, but it missed many red-cylinder contacts. Its top-ranked grasp also contacted less often than the unchanged command. This experiment adds targeted red-cylinder examples and checks candidate ranking again in fresh PyBullet scenes.

The JEPA model and arm controller stay fixed. The contact head stays offline. No physical robot is used.

## Data

Collect 320 training, 80 validation, and 80 test episodes. Every new episode uses the red cylinder and the same yellow pad instruction. The seed ranges are 185000–185319, 185600–185679, and 186200–186279. Use an 80% tipped-object rate, a 45% forced-slip rate, and stronger approach noise than the version 3 collection. Existing version 3 train and validation episodes add variety. Keep the 80 new red-cylinder test episodes separate as the main held-out result.

Collect 40 more red-cylinder scenes for ranking with seeds 187000–187039. Test 25 approach and closure branches from each saved state. Compare the learned score with the old geometric score and the unchanged command. Record the actual finger force after closure. These branches are offline simulator copies. They do not control the task policy.

## Measures and decision

Report average precision, Brier score, precision, recall, and calibration for the targeted test episodes. Also report results on the combined held-out set, with the per-object rows. For ranking, report top-choice contact rate and candidate average precision. Preserve every episode, failed collection, protocol, model hash, and seed list.

Keep the head offline if the fresh red-cylinder test or ranking result is weak or mixed. A new controller comparison needs a clear gain on fresh ranking scenes first. Do not use the 40 ranking scenes to tune the model.

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

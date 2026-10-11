# Train a grasp candidate reranker

## Aim

The version 4 contact head improved contact prediction on red-cylinder scenes, but a fresh ranking check did not beat the unchanged command. This experiment trains a small scoring model to rank grasp candidates from their predicted contact, the older held score, the candidate's offset, its distance from the predicted object position, and whether the object was tipped.

Use 40 candidate-training scenes, 20 validation scenes, and 40 test scenes. Their seed ranges are 188000–188039, 188100–188119, and 188200–188239. Every scene uses the red cylinder. Generate 25 candidate branches per scene and label each branch with the measured finger force after closure.

Use training labels only to fit the reranker. Select its training step on the validation scenes. Keep the test set untouched until the model is frozen. Compare its top candidate with the version 4 contact head, old geometric score, and unchanged command. Report candidate average precision and top-choice contact. These offline branches do not control the arm.

## Decision

Keep the reranker offline if it does not beat the unchanged command on the fresh test scenes or if results vary sharply by scene. A ranking gain does not prove a better grasp, lift, or task completion. Only a clear result should lead to a new frozen controller comparison. Do not use a physical robot.

## Run

Use the locked project environment from the repository root:

```bash
uv run --locked python -m tinyvla.world_contact_reranker collect
uv run --locked python -m tinyvla.world_contact_reranker build
uv run --locked python -m tinyvla.world_contact_reranker train
uv run --locked python -m tinyvla.world_contact_reranker evaluate
```

Outputs go to `artifacts/world-v5/reranker/`. The protocol records source and episode hashes.

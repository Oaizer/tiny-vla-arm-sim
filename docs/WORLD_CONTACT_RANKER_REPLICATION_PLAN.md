# Repeat the V5 contact ranking result

## Aim

The first V5 test used 40 fresh scenes and 1,000 candidate branches. The reranker selected a contacting candidate in 29 scenes, compared with 26 for the unchanged command. It had 7 paired wins and 4 losses. This is a useful signal, but it is not enough to claim a repeatable gain.

Run the frozen V5 model on 40 new red-cylinder scenes, seeds 188240–188279. Keep its weights, inputs, feature scaling, and candidate set fixed. Do not train on these scenes or use them to select settings. Compare the reranker with the unchanged command, the V4 contact head, and the old geometric score.

## Decision

Use the paired top-choice result and candidate average precision to decide whether the V5 ranking gain repeats. This remains a contact-only simulation study. Even a repeated gain does not prove improved lift, retention, or task completion. Do not connect the reranker to the controller from these results alone.

## Run

```bash
uv run --locked python -m tinyvla.world_contact_reranker_replication run
```

The protocol is written before collection. It records source, checkpoint, and episode hashes. Outputs go to `artifacts/world-v5/replication/`.

# V6: guarded world-model proposal with feedback recovery

All tests used the PyBullet arm. No physical robot was used.

## Controller

The feedback policy acts first. If its finger encoder reports an empty grasp, the controller starts its normal bounded retry. During that recovery only, a frozen JEPA model and the V5 contact reranker may propose a grasp offset for the red cylinder. The proposal is limited to the existing 25-point offset set. The reranker falls back to the original grasp if its score margin is below 0.05 or the three contact predictors disagree by more than 0.25. The feedback controller then checks the finger encoder after lift and checks the camera before transport. It retries or stops under its existing rules.

The reranker was trained only for red-cylinder candidates, so it does not change other objects. The test uses one fixed red-cylinder placement instruction.

## Matched test

The frozen test used 40 matched task trials across five conditions, plus 8 missing-object trials. Each task condition used eight fresh seeds. Feedback and hybrid received the same seeds and scene setup.

| Controller | Task successes | Missing-object safe stops | Recovery retries |
|---|---:|---:|---:|
| Feedback | 26/40 | 7/8 | — |
| Hybrid | 26/40 | 7/8 | 17 |

Paired outcomes: 0 wins, 0 losses, and 40 ties for hybrid; exact two-sided sign test p=1.0000.

The reranker made 3 proposals and used its original-grasp fallback 0 times across 3 calls. The complete per-seed outcomes and decisions are in `paired_results.json`; controller logs and selected videos are in `evaluation/`.

## Feature audit

After the test, code inspection found that the V5 reranker feature builder uses the language object's index (red cylinder is channel 1) to select a contact-head output. The contact head uses a different order: red cube, blue cube, green cylinder, red cylinder. Channel 1 therefore contains the blue-cube contact forecast. V6 kept the frozen V5 feature mapping so the ranker received the inputs used during its training and audit.

The V5 tipped-object feature used the simulator's tipped label during training. At runtime, V6 estimated this value from the world model's predicted object orientation. All three accepted proposals had a predicted tipped flag of false. The ranker's runtime inputs therefore do not fully match its training inputs. The controller test is a wiring check, not evidence that the red-cylinder reranker improved grasp selection. Correct the contact channel and use a validated visual tipped-object estimate before the next controller test.

## Limit

This is an initial, small simulation test for one object class. It does not show a reliable general gain. Keep the original feedback controller as the default. A larger, independent matched test is needed before expanding the model to other objects. No real arm was used.

## Reproduce

Run `uv run python -m tinyvla.world_hybrid_experiment`. The command writes `protocol.json` before evaluation and refuses to overwrite an existing run. The protocol records source and checkpoint hashes, settings, and the seed range.

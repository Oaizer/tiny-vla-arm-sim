# V9: bounded retry with the last valid visual plan

All trials used the PyBullet arm. No physical robot was used.

## Method

The hybrid controller used the corrected V7 contact reranker. If the source fell below the visual confidence limit during recovery or the pregrasp recheck, it reused the last valid action for that bounded retry. The original stop behavior remained for a low-confidence first observation. The normal workspace, pad visibility, finger encoder, camera recheck, and retry limits remained active.

## Matched test

The test used 40 paired task trials over five conditions, plus 8 missing-object safety trials.

| Controller | Task successes | Missing-object safe stops |
|---|---:|---:|
| Feedback | 21/40 | 8/8 |
| V9 hybrid | 21/40 | 8/8 |

Paired outcomes: 0 wins, 0 losses, 40 ties for V9; exact two-sided sign test p=1.0000.
The stale-plan fallback ran 30 times. Per-seed traces and videos are in `evaluation/` and `paired_results.json`.

## Limit

This result applies only to one placement task, a small set of simulated seeds, and the current arm model. It does not support real hardware use.

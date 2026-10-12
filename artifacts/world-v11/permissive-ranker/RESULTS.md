# V11: permissive corrected-ranker proposals

All trials used PyBullet. No physical robot was used.

The V7 contact ranker made more top-one contacts than the unchanged candidate on a separate set of 40 scenes. Earlier controller runs used a score-margin and disagreement gate that accepted few proposals. V11 lowered the score-margin limit to zero and raised the disagreement limit to one. Proposals stayed inside the existing 25-candidate set and ran only during recovery. The feedback controller still checked each grasp and kept the same retry budget.

The matched test used 40 task trials, plus 8 missing-object safety trials.

| Controller | Task successes | Missing-object safe stops |
|---|---:|---:|
| Feedback | 24/40 | 8/8 |
| V11 hybrid | 24/40 | 8/8 |

Paired outcomes: 0 wins, 0 losses, 40 ties for V11; exact two-sided sign test p=1.0000.
It made 4 proposals across 4 calls.

Per-seed outcomes, decisions, and trial videos are in `paired_results.json` and `evaluation/`.

# V12: corrected-ranker proposals on the first grasp

All trials used PyBullet. No physical robot was used.

The V7 contact ranker made more top-one contacts than the unchanged candidate on a separate set of 40 scenes. Earlier controller runs used a score-margin and disagreement gate that accepted few proposals. V12 lowered the score-margin limit to zero and raised the disagreement limit to one. Proposals stayed inside the existing 25-candidate set and ran on the first grasp and recovery. The feedback controller still checked each grasp and kept the same retry budget.

The matched test used 40 task trials, plus 8 missing-object safety trials.

| Controller | Task successes | Missing-object safe stops |
|---|---:|---:|
| Feedback | 23/40 | 7/8 |
| V12 hybrid | 23/40 | 7/8 |

Paired outcomes: 2 wins, 2 losses, 36 ties for V12; exact two-sided sign test p=1.0000.
It made 41 proposals across 41 calls.

Per-seed outcomes, decisions, and trial videos are in `paired_results.json` and `evaluation/`.

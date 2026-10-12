# V14 motion comparison

Both controllers used the same V13 ranker, policy, settings, and 40 scene seeds (360000–360039). V14 removed the initial approach probe, return to the parked pose, and second pregrasp camera inference. It uses the first plan from the clear parked camera view. Finger checks, retries, and the destination camera replan after lift remain active.

The original and direct-approach hybrid each completed 21/40 tasks. All 40 paired task outcomes tied. The exact paired sign test was p=1.000. Both controllers stopped safely in all 8 missing-object trials.

The original hybrid made 122 policy inference calls across the 40 task trials. V14 made 89. Mean control time did not improve: 2.06 seconds per task for the original and 2.19 seconds for V14. The removed probe is visible in the matched videos, but fewer inference calls did not make the full simulated task faster.

The videos use the same successful nominal scene, seed 360000. They show the motion change in one example, not average behavior. The detailed logs and frozen source hashes are in `evaluation/`, `original-motion/`, and `protocol.json`.

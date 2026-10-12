# V13: rank grasps by complete simulated task success

All work used PyBullet. No physical arm was used.

On 40 fresh offline scenes, the task-success ranker selected a successful grasp in 12/40 scenes versus 4/40 for the unchanged grasp (8 wins, 0 losses).

On 40 fresh closed-loop matched trials, feedback succeeded 26 times and V13 succeeded 30 times. Paired outcomes were 8 wins, 4 losses, and 28 ties; exact two-sided sign p=0.3877.

Missing-object safe stops: feedback 7/8; V13 7/8.

The ranker uses camera-derived features and a corrected contact channel. It was trained on full simulated pick, lift, and place outcomes. The controller limits changes to the existing 25 grasp offsets, checks each grasp with the finger encoder and camera, and keeps its retry budget.

Per-seed results, decisions, and videos are in `paired_results.json` and `controller-v2/evaluation/`.

# V13 offline task-outcome ranking

The 241-parameter ranker predicts whether each grasp candidate will complete a simulated pick, lift, and place. We trained it on 40 scenes, selected a checkpoint on 20 validation scenes, and checked it on 40 new scenes. Each scene had 25 saved-state branches.

On the fresh test scenes, the ranker chose a candidate that completed the task in 12/40 scenes. The unchanged grasp completed the task in 4/40. Paired results were 8 wins, 0 losses, and 32 ties for the ranker.

This result was a screening test. It used a scripted simulator transport path for branch labels. It did not prove closed-loop controller success. The frozen controller test and independent replication are in `RESULTS.md` and `REPLICATION.md`.

The JSON metric field is named `top1_contact_rate` by a shared metric function. In this V13 file, its value means top-one **task success** because the labels are full-task outcomes. It does not mean contact-only success.

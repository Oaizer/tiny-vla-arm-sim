# World model experiment log

## Plan and data

We used the existing PyBullet arm, trained VLA, and CPU environment. We first recorded 200 pilot episodes. We then extended the dataset to 800 training episodes, 100 validation episodes, and 100 prediction-test episodes. Collection used separate seed ranges. All data is simulated.

Each transition stores a completed command and its duration. This is a change from the initial idea of fixed-rate video steps. It matches the existing servo API and keeps the original controller available for matched tests. The recorder saves actual images and sensor values before and after motion.

## First fit

The first fit tried to predict every object's full position. Validation position error was about 235 mm. An inspection of validation labels found positions from x −4.07 m to x 5.13 m. Some bad grasps had thrown objects far outside the camera view. These failures were valid simulation outcomes, but their exact positions were not observable.

We kept the raw records. We added a derived workspace-presence label and applied pose losses only within the supported workspace. The report also keeps unbounded error. We added explicit landmark and sensor coordinates to the latent state, with a learned residual readout. The first fit's weights and logs remain in the local `attempts/initial_model` folder. That older weight layout needs the earlier architecture and is not a deployment checkpoint.

## Final model training

The fixed CNN supplies spatial features. The trainable state combines 25 direct input values and 71 learned values. Three recurrent predictors learn future states. Training uses a supervised warmup, a fixed-encoder stage, and a JEPA stage with a slowly updated target encoder. Both stages use the same trajectories and simulation labels. They are selected on validation position error.

Prediction tests use 100 separate episodes. We compare actual commands, shuffled commands, and no-change predictions. The learned latent standard deviation is also saved to check for a constant representation. Model weights are held fixed during controller development and the final arm tests.

## Planner development

The first change penalty was too strong. The planner kept the baseline command throughout the completed development trials. Those results are retained in `development/`.

The next setting reduces the change penalty and allows planning only after a failed grasp. This focuses the experiment on recovery. The new development run is in `development-active/`. It compares both model stages on normal, tipped-object, and slip scenes. Final evaluation uses a separate set of seeds.

On those 48 development trials, the existing controller succeeded 33 times, the fixed-encoder planner 32 times, and the JEPA planner 30 times. Both planners changed commands. Neither earned a change to the default controller. We froze these settings and included both stages in a final 720-trial comparison. `FINAL_PROTOCOL.json` records the model hashes and source commit before that comparison.

The early development run used four inference threads. The later run uses one. Small numerical changes can affect simulated contact outcomes. Each comparison uses the same thread count for its baseline and candidate. Timings from concurrent development runs are not used as the final speed result.

## Interface checks

All eight project checks passed before the controller comparison. The three new checks confirm recorded command timing, unchanged servo motion through the sensor wrapper, and a planner interface that works without object-state methods. Full arm outcomes and prediction errors provide the main evidence for model behavior.

The complete dataset has 56,321 executed commands: 45,429 in training, 5,155 in validation, and 5,737 in prediction tests. The raw trajectory files occupy about 410 MB. A separate local data ZIP contains those files. They are not added to Git.

## Evidence to read

- `artifacts/world-v1/RESULTS.md`: the measured result and limits.
- `prediction_results.json`: prediction errors, baselines, and model hashes.
- `model/training.jsonl`: both training stages and validation scores.
- `evaluation/`: each final trial, paired videos, and summaries.
- `manifest.json`: source, model, and trajectory hashes.
- `train_data.json`, `validation_data.json`, `test_data.json`: the episode inventory.

The report must distinguish a working world-model pipeline from a proven gain in robot success. Failed trials stay in the evidence.

## Second experiment: keep the grasp offset consistent

We inspected the version 1 paired outcomes. JEPA won two trials and lost five against feedback. All seven involved cylinders in tipped or slip scenes. We then replayed 25 candidate sequences from the first planning state in each of those seven trials. The selected sequence had a worse measured short-horizon score in all five old losses. Position error was 76.9 mm across the 137 candidate endpoints that stayed in the supported workspace. The repeated baseline branches reproduced their endpoints within one micrometre. These selected cases show a prediction and ranking problem; they do not estimate its frequency across all scenes.

We tested one execution change. The planner selects an offset during open descent and keeps it through closure. The normal controller then handles lift. We held the weights, candidate set, score, and recovery trigger fixed. Nine interface and project tests passed. Commit `4da4b1b` and `artifacts/world-v2/FINAL_PROTOCOL.json` froze the evaluated source and inputs before the new test.

The fresh comparison used 720 trials: three controllers, six conditions, and 40 trials per condition. Feedback completed 150 of 200 tasks. Original JEPA and the consistent-grasp variant each completed 145. Each planner had zero paired wins and five losses against feedback; the two-sided exact paired test gave p=0.0625. Each controller stopped before motion in 39 of 40 missing-object trials. Those missing-object trials are excluded from the task-success denominator.

The consistent-grasp variant reduced planning calls from 146 to 41. Mean task control time was 1.032 seconds for feedback, 1.085 for original JEPA, and 1.081 for the new variant. Fewer planning calls did not produce better task outcomes or a material measured speed gain in this run. The logged `changed_commands` field counts accepted planner choices and excludes the new variant's repeated offset at closure.

Feedback remains the default. The execution change is available only as an experimental variant. The next useful target is contact prediction and candidate ranking, using separate development scenes. Keep these final seeds for reproduction, not for choosing new settings. Read the version 2 report, prediction audit, paired trial records, and output manifest for the full evidence.

## Third experiment: predict direct finger contact

The version 2 planner did not improve task completion. We trained a small contact head using the measured force between the PyBullet fingers and the target object. The version 1 JEPA model stayed frozen. We used 240 training, 80 validation, and 80 test scenes with separate seed ranges. We recorded 13,347, 4,205, and 4,390 servo transitions in those splits. The contact threshold was 0.01 N. The new head has 6,468 parameters.

On held-out test scenes, macro average precision rose from 0.223 with the old geometric held score to 0.585. Macro Brier score fell from 0.138 to 0.068; a constant no-contact score gave 0.118. The head did not help every object. Its red-cylinder recall was 9.3% at a 0.5 threshold, and its red-cube average precision and Brier score were worse than the old score.

We then tested candidate ranking in 40 additional scenes, with 25 simulated grasp branches in each scene. The learned score selected contact in 25 scenes (62.5%). The unchanged command made contact in 27 (67.5%), and the old score selected contact in 21 (52.5%). The head's candidate average precision was 0.778, compared with 0.738 for the old score. The top-choice result is mixed and the sample is small. It does not show improved grasp success or task completion.

Keep this head out of the controller. The first ranking attempt is retained as an incomplete pilot: its selector found only 21 eligible scenes. The corrected audit used fresh seeds and all 40 planned scenes. No controller comparison and no real-arm test used this model. The next simulation work should add targeted red-cylinder contact and slip scenes, then repeat a ranking check on new development seeds. Only a clear ranking gain should lead to a new frozen controller comparison. See [the version 3 report](../artifacts/world-v3/contact/RESULTS.md), its data and model archive in the GitHub release, and `scripts/verify_world_contact.py` for hash and data checks.

## Fourth experiment: target red-cylinder contact

The version 3 head had low red-cylinder recall. We collected 103 targeted red-cylinder training scenes, 24 validation scenes, and 40 fresh test scenes. We reused the version 3 training and validation data to retain examples for the other three objects. The version 1 JEPA model stayed frozen. Most targeted training scenes used a tipped cylinder; about half included the forced-slip action. Collection took longer than expected, so we reduced the original scene count and saved the change in the version 4 plan.

On the 40 fresh red-cylinder test scenes, average precision rose to 0.777 from 0.732 for the version 3 head. Brier score fell from 0.144 to 0.112. Recall at 0.5 rose from 7.2% to 29.8%. Calibration error was still 0.097, and recall remained low. This is better prediction on the target object, but it does not show reliable grasp control.

The first candidate check used 20 red-cylinder scenes and 500 branches. The new head and unchanged command each selected contact in 13 scenes, with matching outcomes in every scene. The old geometric score did so in nine. Average precision across candidates was 0.849 for the new head and 0.769 for the old score. We then froze the head and ran an independent ranking check on 40 new scenes and 1000 branches. The head selected contact in 20 scenes; the unchanged command did so in 22, and the old score in 18. Candidate average precision was 0.742 for the new head and 0.719 for the old score. The independent check did not confirm a top-choice gain over the unchanged command. Both checks measured contact after closure only, not lift, retention, or task success.

Keep the new head offline. The independent ranking check did not beat the unchanged command. Collect more diverse grasp and slip scenes and improve the selection result before planning a controller comparison. No real robot was used. The [version 4 report](../artifacts/world-v4/contact/RESULTS.md), protocols, episode files, and manifest contain the detailed evidence.

## Fifth experiment: rerank grasp candidates

The version 4 head improved red-cylinder contact prediction, but its independent ranking check did not beat the unchanged grasp. We trained a 241-parameter reranker to combine the three contact forecasts, the old geometric scores, each candidate's offset, its offset from the predicted target position, and whether the target was tipped. The frozen JEPA model, version 4 contact head, and arm controller did not change.

We used 40 training scenes, 20 validation scenes, and 40 test scenes, then froze the model before a further 40-scene replication. Each scene had 25 candidate branches. The labels came from measured PyBullet finger force after the close command. Training selected step 100 using validation candidate average precision.

On the first fresh test, the reranker selected contact in 29/40 scenes; the unchanged grasp did so in 26/40. Mean per-scene candidate average precision was 0.816 for the reranker and 0.777 for the unchanged grasp. The paired result was 7 wins and 4 losses, with 29 ties.

The independent replication selected contact in 27/40 scenes, compared with 25/40 for the unchanged grasp. Candidate average precision was 0.832 and 0.801. The paired result was 5 wins and 3 losses, with 32 ties. Together, the two checks show 12 wins, 7 losses, and 61 ties. The exact two-sided sign test gives p=0.359.

This is a consistent but small contact-ranking lead. It does not establish better lift, retention, placement, or task success. Keep the reranker offline. The next step is a larger independent ranking test; consider a frozen controller comparison only if it confirms a clear gain. All scenes are simulated. See `artifacts/world-v5/reranker/RESULTS.md`, both version 5 protocols, the verification script, and the version 5 manifest.

## Sixth experiment: guarded model proposal with feedback recovery

We added a hybrid controller for red-cylinder placement. Feedback acts first. If its finger encoder reports an empty grasp, the controller enters its existing bounded retry. During recovery, a frozen JEPA model and the V5 reranker can propose one small grasp offset. The offset is limited to the existing candidate set. The reranker falls back to the original command when its score margin or predictor agreement fails the fixed gate. The feedback controller still checks the finger encoder after lift, replans from the camera, and retries or stops under its existing rules. The V5 ranker is only trained on red-cylinder candidate data, so the hybrid does not change other objects.

The frozen comparison used 40 matched red-cylinder task trials: eight each in nominal, tipped, forced-slip, moved-object, and changed-dynamics scenes. We also ran eight missing-object checks. Feedback and hybrid each completed 26/40 tasks. All 40 paired task outcomes tied. There were no wins or losses for the hybrid (two-sided exact sign test p=1). The ranker made three proposals across 17 retries. Both controllers stopped safely in 7/8 missing-object checks. The small test shows no task-success gain.

The integration audit found two input limits. V5 training selected contact-head channel 1 for the red-cylinder target, but channel 1 is the blue-cube output because the contact head and language module use different object orders. We preserved the V5 mapping in this integration test. V5 also used a simulator tipped label in training; V6 estimated tipped state from the predicted object orientation. All three accepted proposals had a false tipped flag. The reranker inputs do not fully match its training inputs. Keep this hybrid offline. Correct the contact channel and validate an image-based tipped estimate before another controller test.

The protocol records the source and model hashes, settings, and seed ranges. The per-seed records, selected matched videos, and summary are in `artifacts/world-v6/hybrid-feedback/`. Read the V6 report before using the hybrid. It is an integration test, not a general result.

## Seventh experiment: correct contact features

The V6 audit found that the contact head and language model use different object orders. The V5 ranker had used the blue-cube output when it was meant to use the red-cylinder output. We rebuilt the train and validation features with the correct red-cylinder channel and removed the simulator-only tipped label. We kept the V1 world model and V4 contact head frozen.

On 40 fresh ranking scenes, the corrected ranker selected contact in 29/40 scenes. The unchanged grasp did so in 27/40. This was a small contact-only lead. On a separate 40-trial controller test, feedback and hybrid both completed 25 tasks. All 40 outcomes tied. Better contact choice did not improve the task result.

## Eighth experiment: fine-tune on fallen objects

We fine-tuned the VLA on 240 simulated scenes with a fallen red cylinder and mixed half of each training batch from the original data. A 60-scene held-out set lowered the fallen-scene waypoint error from 7.15 mm to 6.08 mm at the selected checkpoint. Error on the original validation scenes rose from 2.07 mm to 2.76 mm.

On 40 new matched task trials, the base model completed 21 tasks and the adapted model completed 19. The paired result was one win and three losses for adaptation. The waypoint gain did not carry over to task success. Keep the adapted checkpoint out of the default policy.

## Ninth experiment: retry from the last valid visual target

Several tipped and slip trials stopped after the camera gave a low source score during recovery. We added a fallback that can reuse a recent valid action only after the controller has seen the source. A low-confidence first observation still stops. Workspace checks, destination checks, finger checks, and the attempt limit stayed active.

The fallback ran 30 times in 40 matched trials. Feedback and hybrid both completed 21 tasks, and all 40 outcomes tied. Reusing the old target did not rescue a task.

## Tenth experiment: route to the adapted policy

We tried a linear classifier, a small multilayer classifier, and a crop-based image classifier to detect a fallen cylinder. The gates used camera and arm inputs. None read the simulator condition. We kept a 5% limit on false positives for upright scenes.

Fresh tipped-object recall was 5% for the linear gate, 2.5% for the multilayer gate, and 2.5% for the crop classifier. The crop gate had a 12.5% false-positive rate. We did not run a controller test because the visual gates missed too many fallen objects.

## Eleventh and twelfth experiments: change when proposals run

The V11 test removed the score-margin and disagreement gates during bounded recovery. It accepted four proposals but had no task wins or losses: both controllers completed 24/40 tasks.

The V12 test also used proposals on the first grasp. It accepted 41 proposals. Both controllers completed 23/40 tasks, with two wins and two losses for the hybrid. The net result was zero. The more active ranker did not improve the task score.

## Thirteenth experiment: train on full task outcomes

The contact ranker used contact after closure as its label. Contact alone did not say whether a grasp would lift and place the object. We trained a 241-parameter ranker with labels from full simulated pick, lift, and place branches. The train split had 40 scenes, validation had 20, and the fresh offline test had 40. Each scene had 25 grasp candidates. Training and validation used V5 train and validation scenes. The test used new seeds.

The ranker selected a grasp that completed the scripted task in 12/40 fresh scenes. The unchanged grasp succeeded in 4/40. This was a screening result, not a closed-loop result. We then tested the frozen ranker in the feedback controller. It could use the existing candidate set on the first grasp and during recovery. The finger checks, camera checks, and retry limit remained active.

On the primary matched test, feedback completed 26/40 tasks and V13 completed 30/40. There were 8 wins, 4 losses, and 28 ties for V13. We then ran an independent 40-trial replication without changing the ranker. Feedback completed 25 tasks; V13 completed 27. There were 3 wins, 1 loss, and 36 ties. Across both tests, V13 completed 57/80 tasks and feedback completed 51/80. The paired result was 11 wins, 5 losses, and 64 ties; exact two-sided sign p=0.210.

Both tests had a positive net gain. The test is still small, and the paired result is not strong statistical evidence. The current result is a repeatable positive simulation signal, not proof of a general gain. Missing-object safe stops were 15/16 for each controller across the two tests. Keep testing with more seeds and more task conditions before using the ranker beyond this simulated red-cylinder task. No real robot was used.

The V13 ranker, data, videos, protocols, and per-seed results are in `artifacts/world-v13/task-ranker/`. Run `uv run --locked python scripts/verify_world_hybrid_v13.py` to check its hashes and paired counts. The portfolio has selected matched videos from the primary test and replication.

## Fourteenth experiment: plan from the parked camera view

The first V13 camera plan already comes from the parked pose. The old controller then moved toward its predicted grasp point, returned to park, and ran another pregrasp inference. We tested a direct-approach variant that follows the first plan and skips that probe, retreat, and second inference. Finger checks, retries, and the destination camera replan after lift remain active.

We ran the original and direct-approach hybrid on the same 40 new task seeds and the same frozen ranker. Both completed 21/40 tasks; all 40 paired outcomes tied. Both safely stopped in all eight missing-object checks. The direct variant used 89 policy inference calls, down from 122. Mean control time did not improve: it was 2.19 seconds per trial versus 2.06 seconds for the original. The motion clips show the removed probe-and-return on one matched successful scene. This removes the visible back-and-forth without a measured task-success change in this seed set; it does not show faster control.

The current simulator still uses a Panda arm. The selected hardware target is the Seeed reBot B601-RS, which has six arm joints plus a gripper and a different robot model and motor interface. Before any hardware work, the next sim phase must use the B601-RS model, limits, and dynamics, then collect new training data. See the [B601-RS simulation plan](WORLD_MODEL_PLAN.md#target-arm-seeed-rebot-b601-rs).

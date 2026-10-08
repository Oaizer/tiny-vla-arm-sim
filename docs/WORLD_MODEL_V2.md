# Test a consistent grasp offset

## Evidence and hypothesis

The version 1 JEPA planner won two paired trials and lost five against feedback. All seven involved cylinders in tipped or slip scenes. It changed commands in 31 of 240 episodes. The selected offsets often lowered the hand by 11 or 12 mm.

The planner can change the descent, closure, and early lift independently. Its scored sequence can therefore differ from the commands it executes. We will test whether one consistent descent and closure offset reduces harmful changes. The current logs do not prove that this is the cause of the failures.

## One change

Choose the offset during open descent. Keep it for the next closure. Then use the normal lift command. Clear the saved offset if a different command interrupts the grasp. Keep the world model weights, score, candidate set, and recovery trigger fixed.

## Diagnosis

Run `python -m tinyvla.world_audit`. This uses the seven old trials where JEPA and feedback had different outcomes. At the first planner call, save the simulator state. Execute each of the 25 proposed three-command sequences from that state. Compare predicted position and held-object score with simulation labels. Repeat the baseline branch to check that state restoration is repeatable.

This is an offline diagnostic. The tested controllers do not receive the branch outcomes or simulator labels. The audit covers one decision from each selected old trial. It does not measure full-task success for each candidate, and it does not represent all scenes.

## Frozen comparison

Run `python -m tinyvla.world_iteration run`. This saves the protocol before it runs any test. It uses feedback, original JEPA, and the consistent-grasp variant on matching fresh seeds. There are 40 trials per condition: nominal, tipped, slip, object moved, dynamics, and missing. The seed blocks start at 170000 and increase by 1000. There are 720 trials in total.

Report task success across the five task conditions. Report stopping before motion separately for missing objects. Report paired wins and losses, an exact paired test, control time, planner calls, and changed commands. Keep all logs and videos. Do not tune parameters on these seeds.

Keep feedback as the default. One run with one trained model is insufficient to establish a reliable general gain. The new command reuses the version 1 weights and records their hashes, the training manifest hash, source hashes, runtime versions, and settings. It refuses to overwrite a protocol. Use `--root` for a new output folder; a rerun of the same seeds is a reproduction, not a new confirmation.

Read `artifacts/world-v2/RESULTS.md`, `paired_results.json`, `prediction_audit.json`, and `FINAL_PROTOCOL.json` after completion. The interface test checks closure, lift, and interruption behavior. The arm comparison tests the effect on task outcomes.

After both runs, use `python scripts/report_world_audit.py` to write the audit report and explain the shared metric fields. Then run `python scripts/verify_world_v2.py` to check the frozen input hashes, matched trial records, summary counts, and repeated simulator branches. It writes hashes for all version 2 output files.

The log field `changed_commands` counts accepted planner choices. For the consistent-grasp variant, closure also receives the retained offset, but that application does not count as a new planner choice. Use planning-call counts to compare the number of model decisions.

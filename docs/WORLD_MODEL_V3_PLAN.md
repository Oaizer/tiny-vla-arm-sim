# Learn contact from the simulator

## What version 2 showed

The world-model planner did not improve task success. An offline replay found 76.9 mm position error across supported candidate endpoints. The model's existing held-object score also did not describe measured finger contact. The current training target defines “held” by object height, distance from the hand, and finger gap. It does not use a contact sensor.

## This experiment

Record actual PyBullet contact force at finger links 9 and 10 for each of four objects. Record that label after every executed servo command. Use separate seeds for training (180120–180359), validation (181240–181319), and test (182320–182399). The ranges do not overlap earlier world-model controller tests.

Train a small contact head on the saved version 1 JEPA latent predictions. Freeze the version 1 model. Do not put simulator labels into the input features. Fit a per-object probability scale and bias on validation scenes only. Compare average precision, Brier score, calibration error, precision, and recall with the previous geometric held score on the fresh test scenes. Average precision helps show performance when contact is rare. Brier score and calibration error show whether a score can be read as a probability.

Run `python -m tinyvla.world_contact collect` to gather the configured train, validation, and test splits. Run `python -m tinyvla.world_contact train` to fit the contact head and score it on validation and test. Both commands use PyBullet only. The report records the seed lists, source hashes, contact threshold, model hash, calibration, and per-object metrics.

## Decision rule

Do not change the planner in this experiment. The new head must improve average precision and Brier score over the old score on test scenes, without losing all recall for any object. It must also be calibrated well enough for its score to have a useful probability meaning. Report failures and per-object results. If it passes, test candidate ranking on separate development seeds before planning a new, frozen arm comparison. If it fails, keep the head offline and expand the contact data before another attempt.

# V7: correct the reranker features

All tests used PyBullet. No physical arm was used.

## Offline ranking

The feature audit found that the contact head and language model use different object orders. The corrected ranker reads red-cylinder contact output channel 3. It also drops the simulator tipped label from the model input. The training scenes came from the V5 train and validation splits. The test used 40 fresh scenes and 1,000 saved-state branches.

The corrected ranker selected contact in 29/40 scenes. The unchanged grasp did so in 27/40. Paired results were 4 wins, 2 losses, and 34 ties. This is a small offline contact gain. It does not measure a completed task.

## Matched controller test

The controller test used 40 matched task trials across five conditions and eight missing-object checks. Feedback and hybrid each completed 25/40 tasks. Every task outcome tied. Both controllers stopped safely in 8/8 missing-object checks. The ranker made only a few proposals, and none changed task success.

Keep this version out of the default controller. See `paired_results.json`, `evaluation/`, `protocol.json`, and `../corrected-ranker/results.json` for the seed-level record and hashes.

# V8: adapt the vision and action model to fallen objects

All trials used PyBullet. No physical arm was used.

The adapted model was fine-tuned on 240 simulated scenes with a fallen red cylinder. It used a 50:50 mix of those examples and the original training examples. The base model and adapted model received identical fresh evaluation seeds.

| Model | Task successes | Missing-object safe stops |
|---|---:|---:|
| Base | 21/40 | 8/8 |
| Adapted | 19/40 | 8/8 |

Paired outcomes: 1 wins, 3 losses, 36 ties for adaptation.

The complete per-seed outcomes, videos, hashes, and training record are stored beside this report. The test covers one object, one placement instruction, and a small number of simulated scenes.

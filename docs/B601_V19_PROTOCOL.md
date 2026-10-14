# B601 V19 evaluation protocol

V18 improved green-cube placements over V17 but placed fewer blue cubes. V19 tests one change: multiply the training loss for blue-cube instructions by 1.5. All other training data, model size, optimizer, and training schedule stay the same as V18.

Before the new policy is evaluated, reserve 120 fresh randomized scenes: twelve trials in each seed block from 631000 through 640011. These seeds were not used to collect training data or select V19. Evaluate V18, V17, and V19 on the same scenes. Run V18 first and save its teacher records; reuse those exact records for V17 and V19. Do not tune the weight or select a checkpoint from these results.

Compare per-trial cube placements with an exact two-sided sign test. Report full sorts, per-color placements, ties, joint-limit violations, model size, and inference time. A higher aggregate alone is not enough to claim a gain. If the paired result remains uncertain, keep V18 labeled as the best observed policy and report the uncertainty.

All tests use the B601-RS PyBullet model. No physical arm is used.

## Results

V19 used the same data and model as V18, with 1.5× loss weight on blue-cube instructions. The selected checkpoint was epoch 24, with 427,798 parameters. It used 7,152 samples from 232 training episodes. The recorded CPU inference time was 4.73 ms per action; this can vary with machine load.

On the reserved 120 seeds, V19 placed 150/360 cubes and completed 14/120 sorts. V18 placed 130/360 and completed 8/120. The paired cube result was 41 wins, 30 losses, and 49 ties (p = 0.235), so the difference is uncertain. V19 placed 50 red, 55 blue, and 45 green cubes; V18 placed 47 red, 50 blue, and 33 green.

V17 placed 119/360 cubes and completed 7/120 sorts on these same scenes. V19's paired result against V17 was 52 wins, 31 losses, and 37 ties (p = 0.028). This is encouraging evidence, but the comparison with V18 is not conclusive and another fresh evaluation is needed. See [the full comparison](../artifacts/b601-bc-v19-blue-weight/COMPARISON.md).

All three policies had zero joint-limit violations on this test. The independent follow-up is documented in the [V19 confirmation protocol](B601_V19_CONFIRMATION_PROTOCOL.md).

# B601 V19 confirmation protocol

V19 placed more cubes than V18 on 120 fresh matched scenes, but the paired result was uncertain (41 wins, 30 losses, 49 ties; p = 0.235). This follow-up tests whether that result repeats on another set of scenes.

Use seed blocks 641000–650011, with twelve trials per block. These scenes were not used for training or checkpoint selection. Run V18 first for each block and save its teacher outcomes. Run the fixed V19 checkpoint on the same block using those saved teacher outcomes. Do not change either checkpoint or tune V19 from these results.

The primary measure is per-trial cube placements, compared with the exact two-sided sign test that excludes ties. Also report full sorts, color-level placements, ties, and any joint-limit violations. Keep the earlier fresh-set result separate from this confirmation result. A second non-significant result does not prove equivalence; it means this experiment did not establish a difference.

All trials use the B601-RS PyBullet model. No physical arm is used.

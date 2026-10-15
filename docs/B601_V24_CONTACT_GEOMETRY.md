# B601 V24 face-contact geometry test

This simulation-only test changes one setting. V24 raises the camera-corrected descend and close waypoints by 30 mm. V23 uses 20 mm. The checkpoint and policy weights stay fixed.

## Results

The matched test used seeds 678000–678059. V23 placed 129/180 cubes and completed 23/60 full sorts. V24 placed 134/180 cubes and completed 22/60 full sorts. Placement outcomes were 15 V24 wins, 15 losses, and 30 ties (p = 1.00). Full sorts were 12 wins, 13 losses, and 35 ties (p = 1.00). This test does not show a task-success gain.

Among 169 matched objects where both policies made finger contact, V24 had lower corner proximity in 141, higher proximity in 28, and tied in 0. Mean object-level corner proximity changed from 0.4193 to 0.2949 (mean change -0.1244; paired bootstrap 95% interval -0.1567 to -0.0904; sign-flip p = 0.00003). In this audit, lower values mean contact closer to the in-plane middle of a cube side and farther from its corners.

Vertical contact stayed near the top or bottom rim. Contacts within 10 mm of the vertical face center fell from 26/442 to 11/469; mean absolute local contact height changed from 18.71 mm to 19.14 mm. The fixed close waypoint still places the fingertips high on the cube. This suggests that yaw and horizontal centering do not fully set contact height; it does not prove a physical robot cause. Keep V23 as the task-success lead; record V24 as an in-plane grasp-geometry result.

All trials used the B601-RS simulation. Contact points are Bullet simulation measurements, not physical tactile-sensor readings. The corner-proximity measure is a geometric proxy and does not establish safer or more reliable contact on real hardware.

## Clips

The [V23 clip](../portfolio/media/b601-rs-v23-v24-face-contact-seed-678001-v23.mp4) and [V24 clip](../portfolio/media/b601-rs-v23-v24-face-contact-seed-678001-v24.mp4) show the same 3/3 seed.

Raw matched results and per-object contact measurements are in [the portfolio data](../portfolio/b601-v24-contact-geometry.json).

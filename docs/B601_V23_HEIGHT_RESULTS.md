# B601 V23 grasp-height results

## Result

V23 keeps the V19 checkpoint and V22 camera correction. It adds 20 mm to the camera-corrected descend and close TCP targets. It leaves XY, yaw, and delivery actions unchanged. The model stays at 427,798 parameters; this experiment changes one waypoint rule and does not retrain the checkpoint.

The fresh matched test used seeds 676000–676119. V23 placed 260 of 360 cubes and completed 47 of 120 full sorts. V22 placed 226 cubes and completed 31 full sorts on the same scenes. V23 had more placements in 49 scenes, fewer in 22, and tied in 49. The exact paired sign test gives p = 0.00182. For full sorts, V23 won 31 scenes, lost 15, and tied in 74 (p = 0.0259). Neither policy had a joint-limit violation.

## Contact audit

The change improved several contact measures. Mean wrist-to-face yaw error fell from 7.20° to 5.27°. Mean corner proximity fell from 0.449 to 0.434. Objects with no finger contact fell from 20 to 14, and two-finger contacts rose from 335 to 339.

The vertical contact shift was small. Mean absolute contact height fell from 18.95 mm to 18.57 mm; the median stayed at 19.55 mm. Contacts within 10 mm of the face center rose from 30 to 61. Contacts within 5 mm of the top or bottom rim fell from 857/894 (95.9%) to 819/883 (92.8%). The higher waypoint improved task completion, but it did not move most contacts to the center of the face. The result does not show that vertical contact caused all of the task gain.

## Pilot and limits

A 12-scene pilot compared −10 mm, +10 mm, and +20 mm. Their placement totals were 12/36, 24/36, and 27/36; V22 placed 18/36. The +20 mm pilot result was not conclusive (8 wins, 2 losses, 2 ties; p = 0.109). We selected +20 mm for the fresh test before running those seeds. Do not treat the pilot as a separate confirmation.

The [matched seed 676015 V22 video](../portfolio/media/b601-rs-v22-v23-height-seed-676015-v22.mp4) shows one placement. The [V23 clip](../portfolio/media/b601-rs-v22-v23-height-seed-676015-v23.mp4) shows all three. Both clips use a 1280 × 768 frame at 10 fps, with an overhead view and synchronized oblique inset.

All runs use PyBullet and the simulated Seeed reBot B601-RS. No physical arm was used. The [compact evaluation data](../portfolio/b601-v23-height-evaluation.json) includes the pilot, matched test, paired seed results, and contact summaries. Full contact records stay in local `artifacts/` output.

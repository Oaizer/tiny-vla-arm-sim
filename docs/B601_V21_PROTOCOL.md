# B601 V21 protocol

V20 had a higher mean wrist-to-face yaw error than V19 and more objects with no finger contact. Its data set also differed in content and was smaller, so the V20 result cannot isolate the effect of pick height.

V21 tests one training change on the V19 data: multiply the yaw loss by 3 on the descend, close, and lift waypoints. Keep V19's 0.08 m teacher pick height, data, blue-cube weight, model, optimizer, batch size, epoch count, and random seed fixed. Keep the ordinary validation score for checkpoint selection. Do not change the simulator or evaluation policy.

Train with:

```powershell
$pythonExe = Join-Path $env:LOCALAPPDATA 'codex-b601-sim-py311\python.exe'
$env:PYTHONPATH = 'src;scripts'
& $pythonExe scripts\train_b601_bc.py `
  --dataset artifacts\b601-demos-v7-overhead384 `
  --extra-training-dataset artifacts\b601-demos-v15-overhead384-extra `
  --extra-training-dataset artifacts\b601-dagger-v16-expanded-40 `
  --extra-training-dataset artifacts\b601-dagger-v17-intervention-fixed-pilot-10 `
  --extra-training-dataset artifacts\b601-dagger-v17-intervention-fixed-40 `
  --output artifacts\b601-bc-v21-grasp-yaw `
  --action-mode cartesian --spatial-size 8 --epochs 30 `
  --visual-target-features --blue-cube-weight 1.5 `
  --grasp-yaw-weight 3 --seed 20261012
```

Evaluate V19 first on 120 fresh seeds, 671000–671119, and save its teacher records. Evaluate V21 on the same seeds, reusing those records. Enable the grasp audit for both runs. Compare placements with the exact paired sign test; also report full sorts, outcomes by color, wrist-to-face yaw error, corner proximity, finger contact, top or bottom rim contact, and joint-limit violations.

This remains a simulation-only experiment. Do not connect a physical arm.

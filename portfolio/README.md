# Simulation portfolio page

The page compares each model iteration and includes nine selected PyBullet trial clips. Every scene and metric is simulated. The page keeps mixed and negative results beside the gains.

## Preview

From the repository root, start a local web server:

```bash
python -m http.server 8000
```

Open `http://localhost:8000/portfolio/`.

## Add an iteration

Add one entry to `iterations.json`. Include the experiment question, seed counts, the main comparison, and what the result does not prove. Link its tracked results report. Add matched video files under `media/` only when they show a useful comparison and keep their captions tied to the measured trial.

The experiment source, configuration, protocols, summaries, and data manifests are in the repository and its versioned GitHub release assets. The raw episode data are too large for normal Git history.

The B601-RS pilot uses the articulated Seeed URDF with PyBullet physics. Its selected clips and tracked [per-seed data](b601-rs-results.json) appear before the older Panda VLA videos. See [B601-RS simulation notes](../docs/B601_SIMULATION.md) for scope and limits.

## B601-RS challenge record

The current lead policy is V23, a 427,798-parameter V19 model with calibrated camera grasp poses and a higher descend/close waypoint. In 120 matched PyBullet scenes, it placed 260/360 cubes and completed 47/120 full sorts. V22 placed 226 cubes and completed 31 full sorts on the same scenes. The page records trained policies, scripted teachers, and earlier failures as separate results.

Run `PYTHONPATH=src python scripts/demo_b601_sort.py --version v7 --trials 12 --no-video` to repeat the batch. See the [B601-RS challenge log](../docs/B601_SIMULATION.md#randomized-sorting-challenge-series) for the protocol and limits.

"""Write the method, measured results, and file hashes for the world model."""
import hashlib
import json
import platform
import subprocess
from pathlib import Path

import torch


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def report(config, root):
    root = Path(root)
    prediction = json.loads((root / 'prediction_results.json').read_text())
    evaluation = json.loads((root / 'evaluation/summary.json').read_text())
    data = {split: json.loads((root / f'{split}_data.json').read_text()) for split in ['train', 'validation', 'test']}
    lines = ['# Simulation world model results', '',
             'This experiment uses only PyBullet data. The model predicts what will happen after an arm command. A short planner uses those predictions to compare nearby grasp choices.', '',
             '## Data', '', '| Split | Episodes | Executed commands | Seed range |', '|---|---:|---:|---|']
    for split, details in data.items():
        lines.append(f"| {split} | {details['episodes']} | {details['transitions']} | {details['seed_start']}–{details['seed_end']} |")
    lines += ['', 'Each record contains the camera image, sensor features, actual command, command duration, and the next observation. Three observations form the model input. Episodes include varied contact heights, poor approaches, fallen cylinders, drops, and retries. Training labels include simulated object poses. The planner receives only camera, calibration, and arm sensor inputs.', '',
              '## Model', '',
              'The original visual network stays fixed. It reads a 64 × 64 image resized to its original input size. A small encoder reads its spatial features and three recent sensor observations. The state has 71 learned values and 25 direct landmark and sensor values. Three recurrent predictors estimate future states from commands and their durations. A residual readout estimates object pose, workspace presence, held-object score, and arm sensor values.', '',
              'We first train the feature encoder on simulation labels. We then freeze it and train future prediction. In the JEPA phase, we train that encoder with future-feature loss and a slowly updated target encoder. A feature-variance loss and the label losses help prevent a constant encoding. The visual CNN stays fixed in both phases. This is a small JEPA-style hybrid with simulation labels, not a reproduction of V-JEPA or a fully self-supervised model.', '',
              '| Model | Parameters, excluding target encoder | Checkpoint bytes |', '|---|---:|---:|']
    for name, entry in prediction.items():
        lines.append(f"| {name} | {entry['parameters_without_target_encoder']:,} | {entry['bytes']:,} |")
    lines += ['', 'The fixed visual network is additional to these parameter counts. The target encoder is used during training and is saved in the checkpoint. It is not used by the planner.', '',
              '## Prediction tests', '',
              'Errors below are measured on held-out episodes. Position error is RMSE over x, y, and z for objects within the supported workspace: x 0.25–0.80 m, y −0.40–0.40 m, z 0–0.40 m. Some failures throw objects outside the view. Their raw poses stay in the dataset, and the JSON report retains unbounded pose error. Their workspace-presence labels still contribute to training. Many objects remain still. The moving-object measure uses objects within the workspace whose per-axis RMS movement exceeds 1 cm. A lower feature error alone does not prove better robot control.', '',
              '| Model / input | Position error after 1 command, mm | After 3 commands, mm | Moving objects after 3 commands, mm |', '|---|---:|---:|---:|']
    for name, entry in prediction.items():
        for kind in ['normal', 'shuffled_actions']:
            row = entry[kind]
            lines.append(f"| {name} / {kind.replace('_', ' ')} | {row['object_xyz_rmse_mm'][0]:.2f} | {row['object_xyz_rmse_mm'][-1]:.2f} | {row['moving_object_xyz_rmse_mm'][-1]:.2f} |")
        row = entry['normal']
        lines.append(f"| {name} / hold current visual estimate | {row['visual_no_change_xyz_rmse_mm'][0]:.2f} | {row['visual_no_change_xyz_rmse_mm'][-1]:.2f} | — |")
    oracle = prediction['jepa']['normal']['oracle_no_change_xyz_rmse_mm']
    lines += ['', f'For reference, holding the true current object position gives {oracle[0]:.2f} mm after one command and {oracle[-1]:.2f} mm after three. That reference uses simulator state and is not available to the controller.', '',
              'The JSON results also include feature error, a no-change feature prediction, held-object score error, sensor error, sample counts, and checkpoint hashes. Shuffled-action tests keep the observations but assign actions from other samples.', '',
              '## Arm tests', '',
              'Both controllers use the same visual policy, servo controls, scene seeds, and task rules. The first attempt uses the existing controller. After a failed grasp, the planner can change nearby grasp commands. It compares short sequences, executes the first command, and observes again. A penalty discourages large changes and disagreement among the three predictors. Their disagreement is a heuristic, not a calibrated confidence bound.', '',
              '| Scene | Existing feedback success | Fixed-encoder planner | JEPA planner |', '|---|---:|---:|---:|']
    for suite in config['evaluation']['suites']:
        rows = [r for r in evaluation['suites'] if r['suite'] == suite]
        baseline = next(r for r in rows if r['mode'] == 'feedback')
        fixed = next(r for r in rows if r['mode'] == 'fixed')
        candidate = next(r for r in rows if r['mode'] == 'jepa')
        key = 'safe_stops' if suite == 'missing' else 'successes'
        label = 'Missing object: stop before movement' if suite == 'missing' else suite.replace('_', ' ').capitalize()
        lines.append(f"| {label} | {baseline[key]}/{baseline['episodes']} | {fixed[key]}/{fixed['episodes']} | {candidate[key]}/{candidate['episodes']} |")
    totals = {mode: sum(r['successes'] for r in evaluation['suites'] if r['mode'] == mode and r['suite'] != 'missing')
              for mode in ['feedback', 'fixed', 'jepa']}
    trials = sum(r['episodes'] for r in evaluation['suites'] if r['mode'] == 'feedback' and r['suite'] != 'missing')
    lines += ['', f"Across {trials} task trials per controller, the existing controller succeeded {totals['feedback']} times, the fixed-encoder planner {totals['fixed']} times, and the JEPA planner {totals['jepa']} times. Missing-object stops are reported separately."]
    for mode in ['fixed', 'jepa']:
        wins, losses = 0, 0
        for suite in config['evaluation']['suites']:
            if suite == 'missing':
                continue
            folder = root / 'evaluation' / suite
            baseline_rows = [json.loads(line) for line in (folder / 'feedback.jsonl').read_text().splitlines()]
            candidate_rows = [json.loads(line) for line in (folder / f'{mode}.jsonl').read_text().splitlines()]
            for baseline_row, candidate_row in zip(baseline_rows, candidate_rows):
                if baseline_row['seed'] != candidate_row['seed']:
                    raise ValueError('The paired trial seeds do not match.')
                wins += candidate_row['success'] and not baseline_row['success']
                losses += baseline_row['success'] and not candidate_row['success']
        lines += ['', f'The {mode} planner gained {wins} successes and lost {losses} successes on matched trials.']
    if totals['jepa'] <= totals['feedback']:
        lines += ['', 'This run does not show an overall success gain from the JEPA planner. The existing feedback controller remains the default. The world model and planner are available as an experiment.']
    else:
        lines += ['', 'The JEPA planner had a higher success count in this run. The sample is small, and this result needs a larger independent repeat before changing the default controller.']
    planners = [r for r in evaluation['suites'] if r['mode'] != 'feedback' and r['planner_p95_ms'] is not None]
    if planners:
        lines += ['', f"The largest per-suite p95 planner call took {max(r['planner_p95_ms'] for r in planners):.2f} ms. This covers candidate prediction and selection. Camera capture and feature extraction are separate; their largest per-suite p95 was {max(r['feature_p95_ms'] for r in evaluation['suites']):.2f} ms. These timings come from this CPU run."]
    lines += ['', 'Each JSONL file contains all trials, changes to commands, stop reasons, retries, and timing. The first trial in each pair has a video. Confidence intervals and results by object are in the JSON summaries. Forty trials per condition give limited precision.', '',
              '## Limits', '',
              'The model has a short history and a short prediction horizon. The gripper keeps its original orientation. A blocked camera view can hide the object pose. The model can favor a plan that only looks good in its own predictions. It cannot anticipate an external object move before observing it. The planner still relies on scripted task phases and retry rules. These results apply to simulation only.', '',
              '## Repeat the run', '', 'From the project folder, run:', '', '```bash',
              'uv run --locked python -m tinyvla.world_cli run', '```', '',
              'The command collects data, trains both stages, measures prediction, compares controllers, and writes this report. Run each stage separately with `collect`, `train`, `predict`, `evaluate`, or `report`. Use `--development` for development scenes. Training and evaluation write their run files, so use a new `--root` to keep an earlier run.', '',
              'Large trajectory files stay in the local output folder. The repository contains the source, settings, trained models, results, hashes, and small videos. `manifest.json` records the exact files for this run.']
    (root / 'RESULTS.md').write_text('\n'.join(lines) + '\n')
    project = Path(__file__).resolve().parents[2]
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=project, text=True).strip()
    sources = {str(path.relative_to(project)): digest(path) for path in (project / 'src').rglob('*.py')}
    trajectories = {str(path.relative_to(root)): digest(path) for path in sorted((root / 'trajectories').rglob('*.npz'))}
    manifest = {'git_commit': commit, 'config': config, 'python_platform': platform.platform(),
                'torch': torch.__version__, 'sources_sha256': sources, 'trajectories_sha256': trajectories,
                'models_sha256': evaluation['model_hashes'], 'policy_sha256': evaluation['policy_sha256']}
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps({'report': str(root / 'RESULTS.md'), 'trajectory_files': len(trajectories)}), flush=True)

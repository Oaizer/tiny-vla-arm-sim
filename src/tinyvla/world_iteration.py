"""Inspect paired failures, freeze a protocol, and test one planner change."""
import argparse
import hashlib
import json
import math
import platform
import subprocess
from pathlib import Path

import numpy as np
import torch

from .world_evaluate import benchmark


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def paired(directory, mode):
    rows = []
    for path in sorted(Path(directory).glob('*/feedback.jsonl')):
        baseline = {r['seed']: r for r in map(json.loads, path.read_text().splitlines())}
        for row in map(json.loads, path.with_name(mode + '.jsonl').read_text().splitlines()):
            base = baseline[row['seed']]
            decisions = [d for d in row['planning'] if d['accepted']]
            rows.append({'suite': row['suite'], 'seed': row['seed'], 'object': row['object'],
                         'baseline_success': base['success'], 'candidate_success': row['success'],
                         'baseline_status': base['status'], 'candidate_status': row['status'],
                         'changed_commands': len(decisions), 'decisions': decisions})
    wins = [r for r in rows if r['candidate_success'] and not r['baseline_success']]
    losses = [r for r in rows if r['baseline_success'] and not r['candidate_success']]
    n = len(wins) + len(losses)
    p = min(1., 2 * sum(math.comb(n, k) for k in range(min(len(wins), len(losses)) + 1)) / 2**n) if n else 1.
    return {'mode': mode, 'wins': wins, 'losses': losses, 'discordant_exact_p': p,
            'changed_episodes': sum(r['changed_commands'] > 0 for r in rows), 'paired_trials': rows}


def report(root):
    root = Path(root)
    result = json.loads((root / 'evaluation/summary.json').read_text())
    comparisons = [paired(root / 'evaluation', m) for m in ['jepa', 'coherent']]
    (root / 'paired_results.json').write_text(json.dumps(comparisons, indent=2))
    lines = ['# Coherent grasp experiment', '',
             'All trials use PyBullet simulation. The model weights and training data are unchanged.', '',
             '## Change and reason', '',
             'The previous JEPA planner won two paired trials and lost five against feedback. All seven involved cylinders in tipped or slip scenes. It changed commands in 31 episodes. Many selected offsets moved the hand down by about 11 or 12 mm.', '',
             'Code inspection found that descent, closure, and early lift could each get a new offset. The scored three-command sequence was therefore not an execution commitment. This is a plausible fault, not a proven cause of the seven different outcomes. The old logs do not record each proposed command or each predicted state, so they cannot establish prediction accuracy at those decisions.', '',
             'The new variant chooses once during the open descent. It applies the same offset to the next closure, then returns lift control to the existing controller. A command outside this sequence clears the offset. The score, candidate set, weights, and recovery trigger remain fixed. The model still approximates the lift with a short vertical command; it does not predict the full task.', '',
             '## Frozen test', '',
             'The protocol was saved before evaluation. Each controller uses the same 40 seeds per condition. Six conditions and three controllers give 720 trials. The fresh seed blocks start at 170000 and increase by 1000 per condition. No parameter selection uses these results. FINAL_PROTOCOL.json records the source, input hashes, environment, and settings.', '',
             '| Condition | Feedback | Original JEPA | Coherent grasp |',
             '|---|---:|---:|---:|']
    for suite in result['config']['evaluation']['suites']:
        records = {s['mode']: s for s in result['suites'] if s['suite'] == suite}
        lines.append('| ' + suite + ' | ' + ' | '.join(f"{records[m]['successes']}/{records[m]['episodes']}" for m in ['feedback', 'jepa', 'coherent']) + ' |')
    lines += ['', 'The missing-object condition measures stopping, not task completion. Do not add it to the task-success denominator.', '',
              '| Controller | Task success (200 trials) | Stops before motion, missing (40) | Mean control seconds, tasks | Planner calls | Changed commands |',
              '|---|---:|---:|---:|---:|---:|']
    totals = {}
    for mode in ['feedback', 'jepa', 'coherent']:
        tasks = [s for s in result['suites'] if s['mode'] == mode and s['suite'] != 'missing']
        missing = next(s for s in result['suites'] if s['mode'] == mode and s['suite'] == 'missing')
        total = sum(s['successes'] for s in tasks)
        totals[mode] = total
        lines.append(f"| {mode} | {total}/200 | {missing['safe_stops']}/40 | {np.mean([s['mean_control_seconds'] for s in tasks]):.3f} | {sum(s['planner_calls'] for s in tasks)} | {sum(s['changed_commands'] for s in tasks)} |")
    lines += ['', '## Paired outcomes', '']
    for c in comparisons:
        lines.append(f"{c['mode']}: {len(c['wins'])} wins and {len(c['losses'])} losses against feedback; two-sided exact paired test p={c['discordant_exact_p']:.4f}. Commands changed in {c['changed_episodes']} episodes. Full seed and object records are in paired_results.json.")
        lines.append('')
    lines += ['## Decision', '',
              f"Coherent grasp completed {totals['coherent'] - totals['feedback']:+d} more tasks than feedback and {totals['coherent'] - totals['jepa']:+d} more than original JEPA on these 200 task trials.", '',
              'Keep feedback as the default. This single run and a single trained model do not establish a reliable general gain. The planner remains experimental. Future work should check predicted held-object scores against actual simulated outcomes for matched candidate actions, then improve contact prediction if those scores are poorly calibrated. Use new seeds for any later confirmation.', '',
              '## Reproduce', '',
              'Run `python -m tinyvla.world_iteration run` from the project root. Use `--root` for a new output directory. The command refuses to overwrite an existing protocol. It uses the existing version 1 checkpoints and does not retrain them. Run `python -m tinyvla.world_iteration report` to rebuild this report. Training remains available through `python -m tinyvla.world_cli run`; its data and model versions are stored in the original experiment manifest.', '',
              'The interface test checks that closure retains the selected offset, lift receives the original command, and an interrupted sequence clears the offset. Trial logs add requested position, finger command, and command duration to each planner decision. Videos show the first trial in each condition; those trials are not necessarily planner interventions.', '']
    (root / 'RESULTS.md').write_text('\n'.join(lines))
    print(json.dumps({'report': str(root / 'RESULTS.md'), 'task_successes': totals}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['run', 'report'])
    parser.add_argument('--root', default='artifacts/world-v2')
    args = parser.parse_args()
    root = Path(args.root)
    if args.command == 'report':
        report(root)
        return
    root.mkdir(parents=True, exist_ok=True)
    protocol_path = root / 'FINAL_PROTOCOL.json'
    if protocol_path.exists():
        raise FileExistsError('Protocol exists. Select a new output directory.')
    config = json.loads(Path('configs/world_model.json').read_text())
    config['evaluation']['final_seed'] = 170000
    config['evaluation']['episodes'] = 40
    config['evaluation']['inference_threads'] = 1
    config['final_models'] = ['jepa', 'coherent']
    config['model_stages'] = {'coherent': 'jepa'}
    config['model_root'] = 'artifacts/world-v1/model'
    config['planner_variants'] = {'coherent': {'coherent_grasp': True}}
    policy = 'artifacts/model/best.pt'
    paths = [Path(policy), Path(config['model_root']) / 'jepa.pt', Path('configs/project.json'), Path('uv.lock')]
    paths += sorted(Path('src/tinyvla').glob('*.py'))
    paths += sorted(Path('tests').glob('*.py'))
    protocol = {'config': config, 'hashes': {str(p): sha(p) for p in paths},
                'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                'python': platform.python_version(), 'torch': torch.__version__,
                'training_manifest_sha256': sha('artifacts/world-v1/manifest.json'),
                'hypothesis': 'A consistent descent and closure offset reduces harmful planner interventions.',
                'rule': 'One frozen comparison. No tuning on these seeds. Keep feedback as default.'}
    protocol_path.write_text(json.dumps(protocol, indent=2))
    (root / 'v1_failure_analysis.json').write_text(json.dumps([paired('artifacts/world-v1/evaluation', m) for m in ['fixed', 'jepa']], indent=2))
    (root / 'config.json').write_text(json.dumps(config, indent=2))
    torch.set_num_threads(1)
    benchmark(config, root, policy)
    report(root)


if __name__ == '__main__':
    main()

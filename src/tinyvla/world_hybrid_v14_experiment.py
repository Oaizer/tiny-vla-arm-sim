"""Compare the V13 controller with a direct-approach motion variant."""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import torch

from . import world_evaluate
from .world_hybrid_corrected import CorrectedHybridGraspPlanner
from .world_hybrid_v14_direct import DirectApproachAgent, DirectApproachPlanningAgent

ROOT = Path('artifacts/world-v14/direct-approach')
CONFIG = Path('configs/world_hybrid_v14.json')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sign_p(wins, losses):
    n = wins + losses
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, k) for k in range(min(wins, losses) + 1)) / (2**n))


def run():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    cfg = json.loads(CONFIG.read_text())
    cfg['model_stages'] = {'hybrid': 'jepa'}
    cfg['evaluation']['final_seed'] = cfg['ranking']['controller_seed']
    ROOT.mkdir(parents=True, exist_ok=True)
    protocol_path = ROOT / 'protocol.json'
    if protocol_path.exists() or (ROOT / 'evaluation').exists():
        raise FileExistsError('V14 direct-approach evaluation already exists.')
    files = [CONFIG, Path('configs/project.json'), Path('uv.lock'),
             Path('src/tinyvla/world_hybrid_v14_experiment.py'),
             Path('src/tinyvla/world_hybrid_v14_direct.py'),
             Path('src/tinyvla/world_hybrid_v13_controller.py'),
             Path('src/tinyvla/world_hybrid_v13_experiment.py'),
             Path('src/tinyvla/world_hybrid_corrected.py'),
             Path('src/tinyvla/world_hybrid_ranker_v7.py'),
             Path('src/tinyvla/world_evaluate.py'), Path('src/tinyvla/world_planner.py'),
             Path('src/tinyvla/controller.py'), Path('src/tinyvla/world_model.py'),
             Path('src/tinyvla/world_contact_rank.py'), Path('src/tinyvla/world_contact_reranker.py'),
             Path('src/tinyvla/world_contact_train.py'), Path('src/tinyvla/world_data.py'),
             Path('src/tinyvla/evaluate.py'), Path('src/tinyvla/sim.py'), Path('src/tinyvla/language.py'),
             Path('artifacts/model/best.pt'), Path('artifacts/world-v1/model/jepa.pt'),
             Path('artifacts/world-v4/contact/contact-head.pt'),
             Path('artifacts/world-v13/task-ranker/ranker.pt'),
             Path('artifacts/world-v13/task-ranker/protocol.json'),
             Path('artifacts/world-v13/task-ranker/controller_protocol_v2.json')]
    protocol = {
        'version': 1,
        'created_unix': int(time.time()),
        'seed_start': cfg['evaluation']['final_seed'],
        'episodes_per_condition': cfg['evaluation']['episodes'],
        'conditions': cfg['evaluation']['suites'],
        'model_sha256': digest(Path(cfg['hybrid']['reranker'])),
        'change': 'Skip the initial approach-and-return probe and pregrasp replan. Use the initial camera plan directly. Keep retries, finger checks, and post-lift destination camera replan.',
        'source_sha256': {str(path): digest(path) for path in files},
    }
    protocol_path.write_text(json.dumps(protocol, indent=2) + '\n')

    world_evaluate.HybridGraspPlanner = CorrectedHybridGraspPlanner
    world_evaluate.Agent = DirectApproachAgent
    world_evaluate.PlanningAgent = DirectApproachPlanningAgent
    result = world_evaluate.benchmark(cfg, ROOT, 'artifacts/model/best.pt')

    rows = []
    for suite in cfg['evaluation']['suites']:
        folder = ROOT / 'evaluation' / suite
        feedback = {row['seed']: row for row in map(json.loads, (folder / 'feedback.jsonl').read_text().splitlines())}
        hybrid = {row['seed']: row for row in map(json.loads, (folder / 'hybrid.jsonl').read_text().splitlines())}
        if feedback.keys() != hybrid.keys():
            raise ValueError(f'Unmatched seeds for {suite}.')
        for seed in feedback:
            rows.append({'suite': suite, 'seed': seed,
                         'feedback_success': feedback[seed]['success'], 'hybrid_success': hybrid[seed]['success'],
                         'feedback_status': feedback[seed]['status'], 'hybrid_status': hybrid[seed]['status'],
                         'feedback_safe_stop': feedback[seed]['safe_stop'], 'hybrid_safe_stop': hybrid[seed]['safe_stop']})
    task = [row for row in rows if row['suite'] != 'missing']
    wins = sum(row['hybrid_success'] and not row['feedback_success'] for row in task)
    losses = sum(row['feedback_success'] and not row['hybrid_success'] for row in task)
    summary = {
        'task_trials': len(task),
        'feedback_successes': sum(row['feedback_success'] for row in task),
        'direct_approach_successes': sum(row['hybrid_success'] for row in task),
        'wins': wins, 'losses': losses, 'ties': len(task) - wins - losses,
        'exact_two_sided_sign_p': sign_p(wins, losses),
        'missing_trials': sum(row['suite'] == 'missing' for row in rows),
        'feedback_missing_safe_stops': sum(row['suite'] == 'missing' and row['feedback_safe_stop'] for row in rows),
        'direct_approach_missing_safe_stops': sum(row['suite'] == 'missing' and row['hybrid_safe_stop'] for row in rows),
        'protocol_sha256': digest(protocol_path), 'evaluation_seconds': result['seconds'], 'rows': rows,
    }
    (ROOT / 'paired_results.json').write_text(json.dumps(summary, indent=2) + '\n')
    report = [
        '# V14: remove the initial probe-and-return', '',
        'All tests used PyBullet simulation. No physical arm was used.', '',
        'V14 removes the first approach to a predicted waypoint, the return to the parked pose, and the second pregrasp inference. It uses the initial camera plan directly. Empty-gripper checks, retries, and the destination camera replan after lift remain active.', '',
        f"On {summary['task_trials']} fresh matched task trials, the original feedback controller completed {summary['feedback_successes']} tasks. The direct-approach controller completed {summary['direct_approach_successes']}. Paired outcomes were {wins} wins, {losses} losses, and {summary['ties']} ties (exact p={summary['exact_two_sided_sign_p']:.4f}).", '',
        f"Missing-object safe stops: feedback {summary['feedback_missing_safe_stops']}/{summary['missing_trials']}; direct approach {summary['direct_approach_missing_safe_stops']}/{summary['missing_trials']}.", '',
        'The trial video records the first seed for each condition. Per-seed outcomes, videos, and the frozen source protocol are in this folder.',
    ]
    (ROOT / 'RESULTS.md').write_text('\n'.join(report) + '\n')
    print(json.dumps({key: value for key, value in summary.items() if key != 'rows'}, indent=2), flush=True)


if __name__ == '__main__':
    run()


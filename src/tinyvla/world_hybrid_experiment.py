"""Freeze and evaluate the guarded hybrid controller on matched simulations."""
from __future__ import annotations

import hashlib
import json
import math
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from .world_evaluate import benchmark

ROOT = Path('artifacts/world-v6/hybrid-feedback')
CONFIG_PATH = Path('configs/world_hybrid_v6.json')


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_paths():
    return [CONFIG_PATH, Path('configs/project.json'), Path('uv.lock'),
            Path('src/tinyvla/world_hybrid_experiment.py'), Path('src/tinyvla/world_hybrid.py'),
            Path('src/tinyvla/world_evaluate.py'), Path('src/tinyvla/world_planner.py'),
            Path('src/tinyvla/controller.py'), Path('src/tinyvla/world_model.py'),
            Path('src/tinyvla/world_contact_reranker.py'), Path('src/tinyvla/world_contact_rank.py'),
            Path('src/tinyvla/world_contact_train.py'), Path('src/tinyvla/world_data.py'),
            Path('src/tinyvla/evaluate.py'), Path('src/tinyvla/sim.py'),
            Path('src/tinyvla/language.py'), Path('artifacts/model/best.pt'),
            Path('artifacts/world-v1/model/jepa.pt'),
            Path('artifacts/world-v4/contact/contact-head.pt'),
            Path('artifacts/world-v5/reranker/ranker.pt'),
            Path('artifacts/world-v5/reranker/protocol.json')]


def exact_sign_p(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(wins, losses) + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def summarize(root: Path):
    evaluation = root / 'evaluation'
    paired = []
    all_rows = []
    for suite in json.loads(CONFIG_PATH.read_text())['evaluation']['suites']:
        feedback_path = evaluation / suite / 'feedback.jsonl'
        hybrid_path = evaluation / suite / 'hybrid.jsonl'
        feedback = {row['seed']: row for row in map(json.loads, feedback_path.read_text().splitlines())}
        hybrid = {row['seed']: row for row in map(json.loads, hybrid_path.read_text().splitlines())}
        if feedback.keys() != hybrid.keys():
            raise ValueError(f'Unmatched seeds in suite {suite}.')
        for seed in sorted(feedback):
            base, candidate = feedback[seed], hybrid[seed]
            row = {'suite': suite, 'seed': seed, 'feedback_status': base['status'],
                   'hybrid_status': candidate['status'], 'feedback_success': base['success'],
                   'hybrid_success': candidate['success'], 'feedback_safe_stop': base['safe_stop'],
                   'hybrid_safe_stop': candidate['safe_stop'], 'retries': candidate['retries'],
                   'planning': candidate['planning']}
            all_rows.append(row)
            if suite != 'missing':
                paired.append(row)
    wins = sum(row['hybrid_success'] and not row['feedback_success'] for row in paired)
    losses = sum(row['feedback_success'] and not row['hybrid_success'] for row in paired)
    ties = len(paired) - wins - losses
    feedback_successes = sum(row['feedback_success'] for row in paired)
    hybrid_successes = sum(row['hybrid_success'] for row in paired)
    decisions = [decision for row in all_rows for decision in row['planning']]
    accepted = sum(item['accepted'] for item in decisions)
    fallback = sum(item.get('fallback', False) or item.get('reason') == 'unsupported_object' for item in decisions)
    summary = {'task_trials': len(paired), 'task_suites': sorted({row['suite'] for row in paired}),
               'matched_seeds': True, 'feedback_successes': feedback_successes,
               'hybrid_successes': hybrid_successes, 'wins': wins, 'losses': losses,
               'ties': ties, 'exact_two_sided_sign_p': exact_sign_p(wins, losses),
               'missing_object_trials': sum(row['suite'] == 'missing' for row in all_rows),
               'feedback_safe_stops_missing': sum(row['suite'] == 'missing' and row['feedback_safe_stop'] for row in all_rows),
               'hybrid_safe_stops_missing': sum(row['suite'] == 'missing' and row['hybrid_safe_stop'] for row in all_rows),
               'hybrid_retries': sum(row['retries'] for row in all_rows),
               'ranker_decisions': len(decisions), 'ranker_accepted': accepted,
               'ranker_fallbacks': fallback, 'rows': all_rows,
               'interpretation': 'A simulation comparison of one red-cylinder hybrid configuration. The model proposes only after a feedback failure; camera and finger-sensor checks and bounded retries remain active. This is not evidence for other objects or real hardware.'}
    (root / 'paired_results.json').write_text(json.dumps(summary, indent=2) + '\n')
    report = [
        '# V6: guarded world-model proposal with feedback recovery', '',
        'All tests used the PyBullet arm. No physical robot was used.', '',
        '## Controller', '',
        'The feedback policy acts first. If its finger encoder reports an empty grasp, the controller starts its normal bounded retry. During that recovery only, a frozen JEPA model and the V5 contact reranker may propose a grasp offset for the red cylinder. The proposal is limited to the existing 25-point offset set. The reranker falls back to the original grasp if its score margin is below 0.05 or the three contact predictors disagree by more than 0.25. The feedback controller then checks the finger encoder after lift and checks the camera before transport. It retries or stops under its existing rules.', '',
        'The reranker was trained only for red-cylinder candidates, so it does not change other objects. The test uses one fixed red-cylinder placement instruction.', '',
        '## Matched test', '',
        f"The frozen test used {summary['task_trials']} matched task trials across five conditions, plus {summary['missing_object_trials']} missing-object trials. Each task condition used eight fresh seeds. Feedback and hybrid received the same seeds and scene setup.", '',
        '| Controller | Task successes | Missing-object safe stops | Recovery retries |',
        '|---|---:|---:|---:|',
        f"| Feedback | {summary['feedback_successes']}/{summary['task_trials']} | {summary['feedback_safe_stops_missing']}/{summary['missing_object_trials']} | — |",
        f"| Hybrid | {summary['hybrid_successes']}/{summary['task_trials']} | {summary['hybrid_safe_stops_missing']}/{summary['missing_object_trials']} | {summary['hybrid_retries']} |", '',
        f"Paired outcomes: {wins} wins, {losses} losses, and {ties} ties for hybrid; exact two-sided sign test p={summary['exact_two_sided_sign_p']:.4f}.", '',
        f"The reranker made {accepted} proposals and used its original-grasp fallback {fallback} times across {len(decisions)} calls. The complete per-seed outcomes and decisions are in `paired_results.json`; controller logs and selected videos are in `evaluation/`.", '',
        '## Limit', '',
        'This is an initial, small simulation test for one object class. It does not show a reliable general gain. Keep the original feedback controller as the default. A larger, independent matched test is needed before expanding the model to other objects. No real arm was used.', '',
        '## Reproduce', '',
        'Run `uv run python -m tinyvla.world_hybrid_experiment`. The command writes `protocol.json` before evaluation and refuses to overwrite an existing run. The protocol records source and checkpoint hashes, settings, and the seed range.', ''
    ]
    (root / 'RESULTS.md').write_text('\n'.join(report))
    return summary


def run():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    ROOT.mkdir(parents=True, exist_ok=True)
    protocol_path = ROOT / 'protocol.json'
    if protocol_path.exists() or (ROOT / 'evaluation').exists():
        raise FileExistsError('A V6 protocol already exists. Keep it and choose a new output directory for another test.')
    config = json.loads(CONFIG_PATH.read_text())
    settings = config['evaluation']
    protocol = {'version': 1, 'created_unix': int(time.time()),
                'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                'python': platform.python_version(), 'torch': torch.__version__, 'torch_threads': 1,
                'seed_start': settings['final_seed'], 'episodes_per_condition': settings['episodes'],
                'conditions': settings['suites'], 'instruction': config['task_instruction'],
                'hypothesis': 'A bounded world-model grasp proposal during recovery can retain feedback verification and improve task success.',
                'guard': {'recovery_only': True, 'min_rank_score_margin': config['planner']['hybrid_min_score_gain'],
                          'max_contact_predictor_disagreement': config['planner']['hybrid_max_contact_disagreement'],
                          'candidate_offsets_m': 'existing 25-point grasp set, at most 12 mm vertical and 6 mm horizontal',
                          'feedback_checks': ['camera replanning', 'finger encoder after lift', 'bounded retry and stop']},
                'scope': 'Red cylinder only. Fixed placement instruction. Simulation only.',
                'source_sha256': {str(path): digest(path) for path in source_paths()}}
    protocol_path.write_text(json.dumps(protocol, indent=2) + '\n')
    policy = 'artifacts/model/best.pt'
    benchmark(config, ROOT, policy)
    summary = summarize(ROOT)
    summary['protocol_sha256'] = digest(protocol_path)
    summary['evaluation_seconds'] = json.loads((ROOT / 'evaluation/summary.json').read_text())['seconds']
    summary_path = ROOT / 'summary.json'
    summary_path.write_text(json.dumps({key: value for key, value in summary.items() if key != 'rows'}, indent=2) + '\n')
    print(json.dumps({key: value for key, value in summary.items() if key != 'rows'}, indent=2), flush=True)


if __name__ == '__main__':
    run()

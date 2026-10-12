"""Run frozen V7 corrected-ranker and matched controller evaluations."""
from __future__ import annotations

import hashlib
import json
import math
import platform
import subprocess
import time
from pathlib import Path

import torch

from . import world_evaluate
from .world_hybrid_corrected import CorrectedHybridGraspPlanner
from .world_hybrid_ranker_v7 import train_and_test

ROOT = Path('artifacts/world-v7/corrected-hybrid')
CONFIG = Path('configs/world_hybrid_v7.json')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sign_p(wins, losses):
    n = wins + losses
    if not n: return 1.0
    return min(1.0, 2*sum(math.comb(n, k) for k in range(min(wins, losses)+1))/(2**n))


def run():
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    config = json.loads(CONFIG.read_text())
    ROOT.mkdir(parents=True, exist_ok=True)
    protocol_path = ROOT / 'protocol.json'
    if protocol_path.exists() or (ROOT/'evaluation').exists():
        raise FileExistsError('V7 evaluation exists; use a new frozen output directory.')
    sources = [CONFIG, Path('configs/project.json'), Path('uv.lock'),
               Path('src/tinyvla/world_hybrid_v7_experiment.py'), Path('src/tinyvla/world_hybrid_ranker_v7.py'),
               Path('src/tinyvla/world_hybrid_corrected.py'), Path('src/tinyvla/world_contact_targeted.py'),
               Path('src/tinyvla/world_contact_data.py'), Path('src/tinyvla/world_contact_train.py'),
               Path('src/tinyvla/world_contact_rank.py'), Path('src/tinyvla/world_contact_reranker.py'),
               Path('src/tinyvla/world_model.py'), Path('src/tinyvla/world_data.py'), Path('src/tinyvla/sim.py'),
               Path('src/tinyvla/evaluate.py'), Path('src/tinyvla/controller.py'), Path('src/tinyvla/world_evaluate.py'),
               Path('src/tinyvla/language.py'), Path('artifacts/model/best.pt'), Path('artifacts/world-v1/model/jepa.pt'),
               Path('artifacts/world-v4/contact/contact-head.pt'), Path('artifacts/world-v5/reranker/ranker.pt'),
               Path('artifacts/world-v5/reranker/protocol.json'), Path('artifacts/world-v5/reranker/branches_train.npz'),
               Path('artifacts/world-v5/reranker/branches_validation.npz')]
    protocol = {'version': 1, 'created_unix': int(time.time()),
                'source_commit': subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
                'python': platform.python_version(), 'torch': torch.__version__, 'threads': 1,
                'ranking_protocol': 'artifacts/world-v7/corrected-ranker/protocol.json',
                'seed_start': config['evaluation']['final_seed'],
                'episodes_per_condition': config['evaluation']['episodes'],
                'conditions': config['evaluation']['suites'],
                'hypothesis': 'Correct contact-channel mapping and remove unavailable tipped-label leakage to improve feedback recovery task success.',
                'scope': 'Red cylinder, fixed placement instruction, PyBullet simulation only.',
                'source_sha256': {str(p): digest(p) for p in sources}}
    protocol_path.write_text(json.dumps(protocol, indent=2)+'\n')
    train_and_test()
    # Do not expose the corrected reranker to controller trials unless fresh
    # scene results show it can beat the original command at contact ranking.
    rank = json.loads(Path('artifacts/world-v7/corrected-ranker/results.json').read_text())
    if rank['paired_vs_unchanged']['wins'] <= rank['paired_vs_unchanged']['losses']:
        (ROOT/'decision.json').write_text(json.dumps({'controller_evaluation_run': False,
            'reason': 'Corrected ranker did not beat unchanged grasp contact on fresh scenes.',
            'offline_results': rank['paired_vs_unchanged']}, indent=2)+'\n')
        print(json.dumps({'controller_evaluation_run': False, 'offline': rank['paired_vs_unchanged']}), flush=True)
        return
    world_evaluate.HybridGraspPlanner = CorrectedHybridGraspPlanner
    world_evaluate.benchmark(config, ROOT, 'artifacts/model/best.pt')
    # Keep V7 outcomes independently auditable.
    rows=[]
    evaluation=ROOT/'evaluation'
    for suite in config['evaluation']['suites']:
        base={r['seed']:r for r in map(json.loads,(evaluation/suite/'feedback.jsonl').read_text().splitlines())}
        candidate={r['seed']:r for r in map(json.loads,(evaluation/suite/'hybrid.jsonl').read_text().splitlines())}
        if base.keys()!=candidate.keys(): raise ValueError(f'Unmatched seeds in {suite}')
        rows.extend({'suite':suite,'seed':s,'feedback_success':base[s]['success'],'hybrid_success':candidate[s]['success'],
                     'feedback_safe_stop':base[s]['safe_stop'],'hybrid_safe_stop':candidate[s]['safe_stop'],
                     'retries':candidate[s]['retries'],'planning':candidate[s]['planning']} for s in base)
    task=[r for r in rows if r['suite']!='missing']; wins=sum(r['hybrid_success'] and not r['feedback_success'] for r in task)
    losses=sum(r['feedback_success'] and not r['hybrid_success'] for r in task); ties=len(task)-wins-losses
    summary={'task_trials':len(task),'feedback_successes':sum(r['feedback_success'] for r in task),
             'hybrid_successes':sum(r['hybrid_success'] for r in task),'wins':wins,'losses':losses,'ties':ties,
             'exact_two_sided_sign_p':sign_p(wins,losses),'missing_trials':sum(r['suite']=='missing' for r in rows),
             'feedback_missing_safe_stops':sum(r['suite']=='missing' and r['feedback_safe_stop'] for r in rows),
             'hybrid_missing_safe_stops':sum(r['suite']=='missing' and r['hybrid_safe_stop'] for r in rows),
             'protocol_sha256':digest(protocol_path),'ranking_protocol_sha256':digest('artifacts/world-v7/corrected-ranker/protocol.json'),
             'rows':rows}
    (ROOT/'paired_results.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='rows'},indent=2),flush=True)


if __name__=='__main__': run()

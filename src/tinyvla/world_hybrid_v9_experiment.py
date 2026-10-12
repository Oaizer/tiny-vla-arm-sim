"""Freeze and evaluate recovery fallback plus corrected world-model proposal."""
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
from .world_hybrid_recovery_v9 import RecoveryFallbackAgent

ROOT=Path('artifacts/world-v9/recovery-fallback')
CONFIG=Path('configs/world_hybrid_v9.json')


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def exact_sign_p(wins,losses):
    n=wins+losses
    if not n:return 1.0
    return min(1.0,2*sum(math.comb(n,k) for k in range(min(wins,losses)+1))/(2**n))


def run():
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    cfg=json.loads(CONFIG.read_text()); ROOT.mkdir(parents=True,exist_ok=True)
    protocol_path=ROOT/'protocol.json'
    if protocol_path.exists() or (ROOT/'evaluation').exists():
        raise FileExistsError('V9 test exists; keep it immutable and use new seeds for another attempt.')
    files=[CONFIG,Path('configs/project.json'),Path('uv.lock'),Path('src/tinyvla/world_hybrid_v9_experiment.py'),
      Path('src/tinyvla/world_hybrid_recovery_v9.py'),Path('src/tinyvla/world_hybrid_corrected.py'),
      Path('src/tinyvla/world_hybrid_ranker_v7.py'),Path('src/tinyvla/world_evaluate.py'),Path('src/tinyvla/world_planner.py'),
      Path('src/tinyvla/controller.py'),Path('src/tinyvla/world_model.py'),Path('src/tinyvla/world_contact_rank.py'),
      Path('src/tinyvla/world_contact_reranker.py'),Path('src/tinyvla/world_contact_train.py'),Path('src/tinyvla/world_data.py'),
      Path('src/tinyvla/evaluate.py'),Path('src/tinyvla/sim.py'),Path('src/tinyvla/language.py'),
      Path('artifacts/model/best.pt'),Path('artifacts/world-v1/model/jepa.pt'),Path('artifacts/world-v4/contact/contact-head.pt'),
      Path('artifacts/world-v7/corrected-ranker/ranker.pt'),Path('artifacts/world-v7/corrected-ranker/protocol.json'),
      Path('artifacts/world-v7/corrected-ranker/results.json')]
    protocol={'version':1,'created_unix':int(time.time()),'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
      'python':platform.python_version(),'torch':torch.__version__,'threads':1,'seed_start':cfg['evaluation']['final_seed'],
      'episodes_per_condition':cfg['evaluation']['episodes'],'conditions':cfg['evaluation']['suites'],
      'hypothesis':'Using the last valid visual action during a bounded retry can recover when the source is briefly occluded after a failed grasp.',
      'guard':'Initial low-confidence observations still stop. The fallback is available only after a prior valid source observation and only during recovery or pregrasp replanning. Existing workspace, destination, finger verification, retry budget, and stop checks remain active.',
      'scope':'One fixed red-cylinder placement task in PyBullet simulation.',
      'source_sha256':{str(path):digest(path) for path in files}}
    protocol_path.write_text(json.dumps(protocol,indent=2)+'\n')
    world_evaluate.PlanningAgent=RecoveryFallbackAgent
    world_evaluate.HybridGraspPlanner=CorrectedHybridGraspPlanner
    result=world_evaluate.benchmark(cfg,ROOT,'artifacts/model/best.pt')
    rows=[]
    for suite in cfg['evaluation']['suites']:
        evaluation=ROOT/'evaluation'/suite
        baseline={r['seed']:r for r in map(json.loads,(evaluation/'feedback.jsonl').read_text().splitlines())}
        hybrid={r['seed']:r for r in map(json.loads,(evaluation/'hybrid.jsonl').read_text().splitlines())}
        if baseline.keys()!=hybrid.keys():raise ValueError(f'Unmatched seeds for {suite}.')
        for seed in sorted(baseline):
            b,h=baseline[seed],hybrid[seed]
            rows.append({'suite':suite,'seed':seed,'feedback_success':b['success'],'hybrid_success':h['success'],
              'feedback_status':b['status'],'hybrid_status':h['status'],'feedback_safe_stop':b['safe_stop'],
              'hybrid_safe_stop':h['safe_stop'],'retries':h['retries'],
              'fallbacks':sum(item['phase']=='stale_source_fallback' for item in h['trace']),
              'planning':h['planning']})
    task=[row for row in rows if row['suite']!='missing']
    wins=sum(row['hybrid_success'] and not row['feedback_success'] for row in task)
    losses=sum(row['feedback_success'] and not row['hybrid_success'] for row in task)
    ties=len(task)-wins-losses
    summary={'task_trials':len(task),'feedback_successes':sum(row['feedback_success'] for row in task),
      'hybrid_successes':sum(row['hybrid_success'] for row in task),'wins':wins,'losses':losses,'ties':ties,
      'exact_two_sided_sign_p':exact_sign_p(wins,losses),'missing_trials':sum(row['suite']=='missing' for row in rows),
      'feedback_missing_safe_stops':sum(row['suite']=='missing' and row['feedback_safe_stop'] for row in rows),
      'hybrid_missing_safe_stops':sum(row['suite']=='missing' and row['hybrid_safe_stop'] for row in rows),
      'fallback_uses':sum(row['fallbacks'] for row in rows),'rows':rows,'protocol_sha256':digest(protocol_path),
      'evaluation_seconds':result['seconds']}
    (ROOT/'paired_results.json').write_text(json.dumps(summary,indent=2)+'\n')
    report=['# V9: bounded retry with the last valid visual plan','',
      'All trials used the PyBullet arm. No physical robot was used.','',
      '## Method','',
      'The hybrid controller used the corrected V7 contact reranker. If the source fell below the visual confidence limit during recovery or the pregrasp recheck, it reused the last valid action for that bounded retry. The original stop behavior remained for a low-confidence first observation. The normal workspace, pad visibility, finger encoder, camera recheck, and retry limits remained active.','',
      '## Matched test','',
      f"The test used {summary['task_trials']} paired task trials over five conditions, plus {summary['missing_trials']} missing-object safety trials.",'',
      '| Controller | Task successes | Missing-object safe stops |','|---|---:|---:|',
      f"| Feedback | {summary['feedback_successes']}/{summary['task_trials']} | {summary['feedback_missing_safe_stops']}/{summary['missing_trials']} |",
      f"| V9 hybrid | {summary['hybrid_successes']}/{summary['task_trials']} | {summary['hybrid_missing_safe_stops']}/{summary['missing_trials']} |",'',
      f"Paired outcomes: {wins} wins, {losses} losses, {ties} ties for V9; exact two-sided sign test p={summary['exact_two_sided_sign_p']:.4f}.",
      f"The stale-plan fallback ran {summary['fallback_uses']} times. Per-seed traces and videos are in `evaluation/` and `paired_results.json`.",'',
      '## Limit','',
      'This result applies only to one placement task, a small set of simulated seeds, and the current arm model. It does not support real hardware use.']
    (ROOT/'RESULTS.md').write_text('\n'.join(report)+'\n')
    print(json.dumps({key:value for key,value in summary.items() if key!='rows'},indent=2),flush=True)


if __name__=='__main__':run()

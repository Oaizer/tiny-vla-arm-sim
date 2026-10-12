"""Replicate V13 on a second frozen seed range without tuning the ranker."""
from __future__ import annotations

import hashlib,json,math,time
from pathlib import Path
import torch
from . import world_evaluate
from .world_hybrid_corrected import CorrectedHybridGraspPlanner

ROOT=Path('artifacts/world-v13/task-ranker')
CONFIG=Path('configs/world_hybrid_v13.json')
SEED=350000


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sign_p(wins,losses):
    n=wins+losses
    return 1. if n==0 else min(1.,2*sum(math.comb(n,k) for k in range(min(wins,losses)+1))/(2**n))


def run():
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    cfg=json.loads(CONFIG.read_text());cfg['model_stages']={'hybrid':'jepa'};cfg['evaluation']['final_seed']=SEED
    output=ROOT/'replication';protocol_path=ROOT/'replication_protocol.json'
    if protocol_path.exists() or (output/'evaluation').exists():raise FileExistsError('V13 replication already exists.')
    files=[CONFIG,Path('src/tinyvla/world_hybrid_v13_replication.py'),Path('src/tinyvla/world_hybrid_v13_controller.py'),
      Path('src/tinyvla/world_hybrid_v13_experiment.py'),Path('src/tinyvla/world_hybrid_corrected.py'),
      Path('src/tinyvla/world_hybrid_ranker_v7.py'),Path('src/tinyvla/world_evaluate.py'),Path('src/tinyvla/world_planner.py'),
      Path('src/tinyvla/controller.py'),Path('src/tinyvla/world_model.py'),Path('src/tinyvla/world_contact_rank.py'),
      Path('src/tinyvla/world_contact_reranker.py'),Path('src/tinyvla/world_contact_train.py'),Path('src/tinyvla/world_data.py'),
      Path('src/tinyvla/evaluate.py'),Path('src/tinyvla/sim.py'),Path('src/tinyvla/language.py'),Path('uv.lock'),
      Path('artifacts/model/best.pt'),Path('artifacts/world-v1/model/jepa.pt'),Path('artifacts/world-v4/contact/contact-head.pt'),
      ROOT/'ranker.pt',ROOT/'protocol.json',ROOT/'controller_protocol_v2.json',ROOT/'offline_results.json']
    protocol={'version':1,'created_unix':int(time.time()),'seed_start':SEED,
      'episodes_per_condition':cfg['evaluation']['episodes'],'conditions':cfg['evaluation']['suites'],
      'ranker_sha256':digest(ROOT/'ranker.pt'),'primary_test_protocol_sha256':digest(ROOT/'controller_protocol_v2.json'),
      'hypothesis':'The frozen V13 task-success ranker will improve matched closed-loop task success on a second independent seed range.',
      'source_sha256':{str(path):digest(path) for path in files}}
    protocol_path.write_text(json.dumps(protocol,indent=2)+'\n')
    world_evaluate.HybridGraspPlanner=CorrectedHybridGraspPlanner
    output.mkdir(parents=True,exist_ok=True)
    result=world_evaluate.benchmark(cfg,output,'artifacts/model/best.pt')
    rows=[]
    for suite in cfg['evaluation']['suites']:
        folder=output/'evaluation'/suite
        b={r['seed']:r for r in map(json.loads,(folder/'feedback.jsonl').read_text().splitlines())}
        h={r['seed']:r for r in map(json.loads,(folder/'hybrid.jsonl').read_text().splitlines())}
        if b.keys()!=h.keys():raise ValueError(f'Unmatched replication seeds: {suite}')
        for seed in b:
            rows.append({'suite':suite,'seed':seed,'feedback_success':b[seed]['success'],'hybrid_success':h[seed]['success'],
              'feedback_safe_stop':b[seed]['safe_stop'],'hybrid_safe_stop':h[seed]['safe_stop'],
              'feedback_status':b[seed]['status'],'hybrid_status':h[seed]['status']})
    task=[r for r in rows if r['suite']!='missing'];wins=sum(r['hybrid_success'] and not r['feedback_success'] for r in task)
    losses=sum(r['feedback_success'] and not r['hybrid_success'] for r in task);ties=len(task)-wins-losses
    replication={'task_trials':len(task),'feedback_successes':sum(r['feedback_success'] for r in task),
      'hybrid_successes':sum(r['hybrid_success'] for r in task),'wins':wins,'losses':losses,'ties':ties,
      'exact_two_sided_sign_p':sign_p(wins,losses),'missing_trials':sum(r['suite']=='missing' for r in rows),
      'feedback_missing_safe_stops':sum(r['suite']=='missing' and r['feedback_safe_stop'] for r in rows),
      'hybrid_missing_safe_stops':sum(r['suite']=='missing' and r['hybrid_safe_stop'] for r in rows),
      'protocol_sha256':digest(protocol_path),'evaluation_seconds':result['seconds'],'rows':rows}
    first=json.loads((ROOT/'paired_results.json').read_text())
    combined_wins=first['wins']+wins;combined_losses=first['losses']+losses;combined_n=first['task_trials']+len(task)
    combined={'task_trials':combined_n,'feedback_successes':first['feedback_successes']+replication['feedback_successes'],
      'hybrid_successes':first['hybrid_successes']+replication['hybrid_successes'],'wins':combined_wins,'losses':combined_losses,
      'ties':combined_n-combined_wins-combined_losses,'exact_two_sided_sign_p':sign_p(combined_wins,combined_losses),
      'primary_seed_range':[340000,340039],'replication_seed_range':[SEED,SEED+39]}
    (ROOT/'replication_results.json').write_text(json.dumps(replication,indent=2)+'\n')
    (ROOT/'combined_results.json').write_text(json.dumps(combined,indent=2)+'\n')
    (ROOT/'REPLICATION.md').write_text('\n'.join(['# V13 independent controller replication','',
      'The task-success ranker and proposal settings stayed frozen. This replication used a new seed range and the same matched feedback comparison.','',
      f"On the primary test, V13 had {first['hybrid_successes']}/{first['task_trials']} successes versus {first['feedback_successes']}/{first['task_trials']} for feedback ({first['wins']} wins, {first['losses']} losses).",'',
      f"On the independent replication, V13 had {replication['hybrid_successes']}/{replication['task_trials']} versus {replication['feedback_successes']}/{replication['task_trials']} ({wins} wins, {losses} losses).",'',
      f"Across both tests: V13 {combined['hybrid_successes']}/{combined['task_trials']}, feedback {combined['feedback_successes']}/{combined['task_trials']}; paired {combined_wins} wins and {combined_losses} losses, exact two-sided sign p={combined['exact_two_sided_sign_p']:.4f}.",'',
      f"Missing-object safe stops in replication: feedback {replication['feedback_missing_safe_stops']}/{replication['missing_trials']}; V13 {replication['hybrid_missing_safe_stops']}/{replication['missing_trials']}.",'',
      'See `replication_results.json` and `combined_results.json` for per-seed data.'])+'\n')
    print(json.dumps({'replication':{k:v for k,v in replication.items() if k!='rows'},'combined':combined},indent=2),flush=True)


if __name__=='__main__':run()

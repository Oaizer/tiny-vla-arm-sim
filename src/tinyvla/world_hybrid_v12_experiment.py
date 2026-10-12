"""Evaluate the corrected ranker on the first grasp and during recovery."""
from __future__ import annotations

import hashlib,json,math,time
from pathlib import Path
import torch
from . import world_evaluate
from .world_hybrid_corrected import CorrectedHybridGraspPlanner

ROOT=Path('artifacts/world-v12/first-grasp-ranker')
CONFIG=Path('configs/world_hybrid_v12.json')


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sign_p(wins,losses):
    n=wins+losses
    return 1. if n==0 else min(1.,2*sum(math.comb(n,k) for k in range(min(wins,losses)+1))/(2**n))


def run():
    torch.set_num_threads(1);torch.set_num_interop_threads(1);cfg=json.loads(CONFIG.read_text())
    ROOT.mkdir(parents=True,exist_ok=True);protocol_path=ROOT/'protocol.json'
    if protocol_path.exists() or (ROOT/'evaluation').exists():raise FileExistsError('V12 result path already exists.')
    files=[CONFIG,Path('configs/project.json'),Path('uv.lock'),Path('src/tinyvla/world_hybrid_v12_experiment.py'),
      Path('src/tinyvla/world_hybrid_corrected.py'),Path('src/tinyvla/world_hybrid_ranker_v7.py'),
      Path('src/tinyvla/world_evaluate.py'),Path('src/tinyvla/world_planner.py'),Path('src/tinyvla/controller.py'),
      Path('src/tinyvla/world_model.py'),Path('src/tinyvla/world_contact_rank.py'),Path('src/tinyvla/world_contact_reranker.py'),
      Path('src/tinyvla/world_contact_train.py'),Path('src/tinyvla/world_data.py'),Path('src/tinyvla/evaluate.py'),
      Path('src/tinyvla/sim.py'),Path('src/tinyvla/language.py'),Path('artifacts/model/best.pt'),
      Path('artifacts/world-v1/model/jepa.pt'),Path('artifacts/world-v4/contact/contact-head.pt'),
      Path('artifacts/world-v7/corrected-ranker/ranker.pt'),Path('artifacts/world-v7/corrected-ranker/protocol.json'),
      Path('artifacts/world-v7/corrected-ranker/results.json')]
    protocol={'version':1,'created_unix':int(time.time()),'seed_start':cfg['evaluation']['final_seed'],
      'episodes_per_condition':cfg['evaluation']['episodes'],'conditions':cfg['evaluation']['suites'],
      'hypothesis':'Applying the corrected reranker to the first grasp as well as retries may convert its fresh offline contact-ranking gain into task success.',
      'proposal_gate':{'minimum_score_margin':cfg['planner']['hybrid_min_score_gain'],
                       'maximum_contact_disagreement':cfg['planner']['hybrid_max_contact_disagreement'],
                       'scope':'First grasp and recovery; bounded existing candidate set; feedback checks and retries remain active.'},
      'source_sha256':{str(path):digest(path) for path in files}}
    protocol_path.write_text(json.dumps(protocol,indent=2)+'\n')
    world_evaluate.HybridGraspPlanner=CorrectedHybridGraspPlanner
    result=world_evaluate.benchmark(cfg,ROOT,'artifacts/model/best.pt')
    rows=[]
    for suite in cfg['evaluation']['suites']:
        folder=ROOT/'evaluation'/suite
        b={r['seed']:r for r in map(json.loads,(folder/'feedback.jsonl').read_text().splitlines())}
        h={r['seed']:r for r in map(json.loads,(folder/'hybrid.jsonl').read_text().splitlines())}
        if b.keys()!=h.keys():raise ValueError(f'Unmatched seeds for {suite}.')
        for seed in b:
            rows.append({'suite':suite,'seed':seed,'feedback_success':b[seed]['success'],'hybrid_success':h[seed]['success'],
              'feedback_status':b[seed]['status'],'hybrid_status':h[seed]['status'],'feedback_safe_stop':b[seed]['safe_stop'],
              'hybrid_safe_stop':h[seed]['safe_stop'],'planning':h[seed]['planning']})
    task=[r for r in rows if r['suite']!='missing'];wins=sum(r['hybrid_success'] and not r['feedback_success'] for r in task)
    losses=sum(r['feedback_success'] and not r['hybrid_success'] for r in task);ties=len(task)-wins-losses
    decisions=[d for r in rows for d in r['planning']]
    summary={'task_trials':len(task),'feedback_successes':sum(r['feedback_success'] for r in task),
      'hybrid_successes':sum(r['hybrid_success'] for r in task),'wins':wins,'losses':losses,'ties':ties,
      'exact_two_sided_sign_p':sign_p(wins,losses),'missing_trials':sum(r['suite']=='missing' for r in rows),
      'feedback_missing_safe_stops':sum(r['suite']=='missing' and r['feedback_safe_stop'] for r in rows),
      'hybrid_missing_safe_stops':sum(r['suite']=='missing' and r['hybrid_safe_stop'] for r in rows),
      'planner_calls':len(decisions),'accepted_proposals':sum(d.get('accepted',False) for d in decisions),
      'protocol_sha256':digest(protocol_path),'evaluation_seconds':result['seconds'],'rows':rows}
    (ROOT/'paired_results.json').write_text(json.dumps(summary,indent=2)+'\n')
    (ROOT/'RESULTS.md').write_text('\n'.join(['# V12: corrected-ranker proposals on the first grasp','',
      'All trials used PyBullet. No physical robot was used.','',
      'The V7 contact ranker made more top-one contacts than the unchanged candidate on a separate set of 40 scenes. Earlier controller runs used a score-margin and disagreement gate that accepted few proposals. V12 lowered the score-margin limit to zero and raised the disagreement limit to one. Proposals stayed inside the existing 25-candidate set and ran on the first grasp and recovery. The feedback controller still checked each grasp and kept the same retry budget.','',
      f"The matched test used {summary['task_trials']} task trials, plus {summary['missing_trials']} missing-object safety trials.",'',
      '| Controller | Task successes | Missing-object safe stops |','|---|---:|---:|',
      f"| Feedback | {summary['feedback_successes']}/{summary['task_trials']} | {summary['feedback_missing_safe_stops']}/{summary['missing_trials']} |",
      f"| V12 hybrid | {summary['hybrid_successes']}/{summary['task_trials']} | {summary['hybrid_missing_safe_stops']}/{summary['missing_trials']} |",'',
      f"Paired outcomes: {wins} wins, {losses} losses, {ties} ties for V12; exact two-sided sign test p={summary['exact_two_sided_sign_p']:.4f}.",
      f"It made {summary['accepted_proposals']} proposals across {summary['planner_calls']} calls.",'',
      'Per-seed outcomes, decisions, and trial videos are in `paired_results.json` and `evaluation/`.'])+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='rows'},indent=2),flush=True)


if __name__=='__main__':run()

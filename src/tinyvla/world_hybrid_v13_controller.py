"""Run the V13 task-success ranker in the matched feedback controller test."""
from __future__ import annotations

import hashlib,json,math,time
from pathlib import Path
import torch
from . import world_evaluate
from .world_hybrid_corrected import CorrectedHybridGraspPlanner

ROOT=Path('artifacts/world-v13/task-ranker')
CONFIG=Path('configs/world_hybrid_v13.json')


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sign_p(wins,losses):
    n=wins+losses
    return 1. if n==0 else min(1.,2*sum(math.comb(n,k) for k in range(min(wins,losses)+1))/(2**n))


def run():
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    cfg=json.loads(CONFIG.read_text());cfg['model_stages']={'hybrid':'jepa'}
    cfg['evaluation']['final_seed']=cfg['ranking']['controller_seed']
    ROOT.mkdir(parents=True,exist_ok=True);protocol_path=ROOT/'controller_protocol_v2.json'
    if protocol_path.exists() or (ROOT/'controller-v2/evaluation').exists():raise FileExistsError('V13 controller test already exists.')
    files=[CONFIG,Path('configs/project.json'),Path('uv.lock'),Path('src/tinyvla/world_hybrid_v13_controller.py'),
      Path('src/tinyvla/world_hybrid_v13_experiment.py'),Path('src/tinyvla/world_hybrid_corrected.py'),
      Path('src/tinyvla/world_hybrid_ranker_v7.py'),Path('src/tinyvla/world_evaluate.py'),Path('src/tinyvla/world_planner.py'),
      Path('src/tinyvla/controller.py'),Path('src/tinyvla/world_model.py'),Path('src/tinyvla/world_contact_rank.py'),
      Path('src/tinyvla/world_contact_reranker.py'),Path('src/tinyvla/world_contact_train.py'),Path('src/tinyvla/world_data.py'),
      Path('src/tinyvla/evaluate.py'),Path('src/tinyvla/sim.py'),Path('src/tinyvla/language.py'),Path('artifacts/model/best.pt'),
      Path('artifacts/world-v1/model/jepa.pt'),Path('artifacts/world-v4/contact/contact-head.pt'),ROOT/'ranker.pt',
      ROOT/'protocol.json',ROOT/'offline_results.json',ROOT/'manifest.json']
    protocol={'version':1,'created_unix':int(time.time()),'seed_start':cfg['evaluation']['final_seed'],
      'episodes_per_condition':cfg['evaluation']['episodes'],'conditions':cfg['evaluation']['suites'],
      'offline_model_sha256':digest(ROOT/'ranker.pt'),'offline_protocol_sha256':digest(ROOT/'protocol.json'),
      'hypothesis':'The task-success ranker will improve closed-loop placement success on fresh matched seeds.',
      'proposal_scope':'First grasp and bounded recovery using the existing 25 candidate offsets; feedback verification and retry limits remain active.',
      'source_sha256':{str(path):digest(path) for path in files}}
    protocol_path.write_text(json.dumps(protocol,indent=2)+'\n')
    world_evaluate.HybridGraspPlanner=CorrectedHybridGraspPlanner
    (ROOT/'controller-v2').mkdir(parents=True,exist_ok=True)
    result=world_evaluate.benchmark(cfg,ROOT/'controller-v2','artifacts/model/best.pt')
    rows=[]
    for suite in cfg['evaluation']['suites']:
        folder=ROOT/'controller-v2/evaluation'/suite
        b={r['seed']:r for r in map(json.loads,(folder/'feedback.jsonl').read_text().splitlines())}
        h={r['seed']:r for r in map(json.loads,(folder/'hybrid.jsonl').read_text().splitlines())}
        if b.keys()!=h.keys():raise ValueError(f'Unmatched seeds for {suite}.')
        for seed in b:
            rows.append({'suite':suite,'seed':seed,'feedback_success':b[seed]['success'],'hybrid_success':h[seed]['success'],
              'feedback_status':b[seed]['status'],'hybrid_status':h[seed]['status'],'feedback_safe_stop':b[seed]['safe_stop'],
              'hybrid_safe_stop':h[seed]['safe_stop'],'planning':h[seed]['planning']})
    task=[r for r in rows if r['suite']!='missing'];wins=sum(r['hybrid_success'] and not r['feedback_success'] for r in task)
    losses=sum(r['feedback_success'] and not r['hybrid_success'] for r in task);ties=len(task)-wins-losses
    summary={'task_trials':len(task),'feedback_successes':sum(r['feedback_success'] for r in task),
      'hybrid_successes':sum(r['hybrid_success'] for r in task),'wins':wins,'losses':losses,'ties':ties,
      'exact_two_sided_sign_p':sign_p(wins,losses),'missing_trials':sum(r['suite']=='missing' for r in rows),
      'feedback_missing_safe_stops':sum(r['suite']=='missing' and r['feedback_safe_stop'] for r in rows),
      'hybrid_missing_safe_stops':sum(r['suite']=='missing' and r['hybrid_safe_stop'] for r in rows),
      'protocol_sha256':digest(protocol_path),'evaluation_seconds':result['seconds'],'rows':rows}
    (ROOT/'paired_results.json').write_text(json.dumps(summary,indent=2)+'\n')
    offline=json.loads((ROOT/'offline_results.json').read_text())['paired_vs_unchanged']
    report=['# V13: rank grasps by complete simulated task success','',
      'All work used PyBullet. No physical arm was used.','',
      f"On 40 fresh offline scenes, the task-success ranker selected a successful grasp in {offline['wins']+sum(r['ranker_success'] and r['unchanged_success'] for r in json.loads((ROOT/'offline_results.json').read_text())['scene_rows'])}/40 scenes versus {sum(r['unchanged_success'] for r in json.loads((ROOT/'offline_results.json').read_text())['scene_rows'])}/40 for the unchanged grasp ({offline['wins']} wins, {offline['losses']} losses).",'',
      f"On {summary['task_trials']} fresh closed-loop matched trials, feedback succeeded {summary['feedback_successes']} times and V13 succeeded {summary['hybrid_successes']} times. Paired outcomes were {wins} wins, {losses} losses, and {ties} ties; exact two-sided sign p={summary['exact_two_sided_sign_p']:.4f}.",'',
      f"Missing-object safe stops: feedback {summary['feedback_missing_safe_stops']}/{summary['missing_trials']}; V13 {summary['hybrid_missing_safe_stops']}/{summary['missing_trials']}.",'',
      'The ranker uses camera-derived features and a corrected contact channel. It was trained on full simulated pick, lift, and place outcomes. The controller limits changes to the existing 25 grasp offsets, checks each grasp with the finger encoder and camera, and keeps its retry budget.','',
      'Per-seed results, decisions, and videos are in `paired_results.json` and `controller-v2/evaluation/`.']
    (ROOT/'RESULTS.md').write_text('\n'.join(report)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='rows'},indent=2),flush=True)


if __name__=='__main__':run()

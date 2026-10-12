"""Replay two frozen V13 matched seeds for the portfolio videos."""
from __future__ import annotations

import hashlib,json,time
from pathlib import Path

import torch
from tinyvla import world_evaluate
from tinyvla.evaluate import load_policy
from tinyvla.world_hybrid_corrected import CorrectedHybridGraspPlanner

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'artifacts/world-v13/task-ranker'
SEEDS=[341000,351001]


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    protocol_path=RUN/'selected_replay_protocol.json'
    if protocol_path.exists():raise FileExistsError('V13 selected replay record already exists.')
    sources=[Path('scripts/export_world_hybrid_v13_replays.py'),Path('configs/world_hybrid_v13.json'),
      Path('src/tinyvla/world_hybrid_corrected.py'),Path('src/tinyvla/world_hybrid_ranker_v7.py'),
      Path('src/tinyvla/world_evaluate.py'),Path('src/tinyvla/world_planner.py'),Path('src/tinyvla/controller.py'),
      Path('src/tinyvla/world_model.py'),Path('src/tinyvla/world_contact_rank.py'),Path('src/tinyvla/world_contact_reranker.py'),
      Path('src/tinyvla/world_contact_train.py'),Path('src/tinyvla/world_data.py'),Path('src/tinyvla/evaluate.py'),
      Path('src/tinyvla/sim.py'),Path('src/tinyvla/language.py'),Path('artifacts/model/best.pt'),
      Path('artifacts/world-v1/model/jepa.pt'),Path('artifacts/world-v4/contact/contact-head.pt'),RUN/'ranker.pt']
    protocol={'version':1,'created_unix':int(time.time()),'purpose':'Recreate selected clips from already evaluated seeds; these replays do not add test trials.',
      'seeds':SEEDS,'ranker_sha256':digest(RUN/'ranker.pt'),
      'source_sha256':{str(path):digest(ROOT/path) for path in sources}}
    protocol_path.write_text(json.dumps(protocol,indent=2)+'\n')
    cfg={'task_instruction':'put the red cylinder on the yellow pad','model_root':'artifacts/world-v1/model',
      'model_stages':{'hybrid':'jepa'},'hybrid':{'contact_model':'artifacts/world-v4/contact/contact-head.pt',
      'reranker':'artifacts/world-v13/task-ranker/ranker.pt'},
      'planner':{'recovery_only':False,'coherent_grasp':True,'hybrid_min_score_gain':0.,'hybrid_max_contact_disagreement':1.},
      'evaluation':{'episodes':1,'final_seed':0,'inference_threads':1,'suites':['tipped']},'final_models':['hybrid']}
    world_evaluate.HybridGraspPlanner=CorrectedHybridGraspPlanner
    records=[]
    for seed in SEEDS:
        cfg['evaluation']['final_seed']=seed;root=RUN/'selected-replays'/f'seed-{seed}';root.mkdir(parents=True,exist_ok=True)
        result=world_evaluate.benchmark(cfg,root,'artifacts/model/best.pt')
        folder=root/'evaluation/tipped'
        feedback=json.loads((folder/'feedback.jsonl').read_text());hybrid=json.loads((folder/'hybrid.jsonl').read_text())
        if feedback['seed']!=seed or hybrid['seed']!=seed:raise ValueError('Replay seed mismatch.')
        records.append({'seed':seed,'feedback_success':feedback['success'],'hybrid_success':hybrid['success'],
          'feedback_status':feedback['status'],'hybrid_status':hybrid['status'],
          'feedback_video_sha256':digest(folder/'feedback.mp4'),'hybrid_video_sha256':digest(folder/'hybrid.mp4'),
          'seconds':result['seconds']})
    (RUN/'selected_replays.json').write_text(json.dumps({'protocol_sha256':digest(protocol_path),'rows':records},indent=2)+'\n')
    print(json.dumps(records,indent=2),flush=True)


if __name__=='__main__':main()

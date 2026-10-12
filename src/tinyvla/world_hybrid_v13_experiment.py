"""Train a ranker on complete task success, then test it on fresh scenes."""
from __future__ import annotations

import hashlib,json,time,math
from pathlib import Path
import numpy as np
import pybullet as p
import torch
from torch import nn

from . import world_evaluate,world_hybrid_ranker_v7 as corrected
from . import world_contact_targeted as targeted
from .evaluate import load_policy
from .language import parse
from .sim import ArmSim
from .world_contact_rank import candidates,average_precision
from .world_contact_reranker import Ranker,scene_metrics
from .world_contact_train import ContactHead
from .world_data import tip_object
from .world_model import load_world
from .world_hybrid_corrected import CorrectedHybridGraspPlanner

ROOT=Path('artifacts/world-v13/task-ranker')
CONFIG=Path('configs/world_hybrid_v13.json')
V5=Path('artifacts/world-v5/reranker')


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def execute_task_branch(sim,task,point,teacher):
    sim.move(point,.04,150)
    sim.move(point,0.,190)
    sim.move(point+[0,0,.03],0.,24)
    sim.move(point+[0,0,.24],0.,150)
    for action in teacher[4:]:sim.move(action[:3],action[3],150)
    return bool(sim.score(task)['success'])


def collect_full_success_labels(episodes,split,cfg):
    branch_labels={};audit={};sim=ArmSim()
    try:
        for path in sorted(episodes.glob('*.npz'),key=lambda item:int(item.stem)):
            with np.load(path) as stored:
                actions=stored['actions'].astype(np.float32);features=stored['features'].astype(np.float32);seed=int(stored['seed'])
            metadata=json.loads(path.with_suffix('.json').read_text());task=parse(metadata['instruction'])
            choice=next((i for i in range(1,len(actions)-3) if actions[i,2]<=.055 and actions[i,3]>.02),None)
            if choice is None:raise ValueError(f'No grasp command in seed {seed}.')
            sim.reset(seed)
            if metadata['tipped']:tip_object(sim,task)
            for action in actions[:choice]:sim.move(action[:3],action[3],round(float(action[4])*240))
            replay_error=float(np.max(np.abs((sim.state()-corrected.STATE_OFFSET)/corrected.STATE_SCALE-features[choice,-4:])))
            if replay_error>.005:raise ValueError(f'Saved branch state mismatch {replay_error:.5f} at {seed}.')
            teacher=sim.teacher(task)
            state_id=p.saveState();ticks=sim.ticks;commands=sim.command_count;heights=sim.max_heights.copy()
            labels=[]
            try:
                for point in candidates(actions[choice,:3].copy()):
                    p.restoreState(state_id);sim.ticks=ticks;sim.command_count=commands;sim.max_heights=heights.copy()
                    labels.append(execute_task_branch(sim,task,point,teacher))
            finally:p.removeState(state_id)
            branch_labels[seed]=np.asarray(labels,np.float32)
            audit[seed]={'full_task_success_candidates':int(sum(labels)),'state_error_max':replay_error,
                         'tipped_training_label_omitted':True}
            print(json.dumps({'split':split,'seed':seed,**audit[seed]}),flush=True)
    finally:sim.close()
    return branch_labels,audit


def build_split_features(episodes,labels):
    base,_=load_world('artifacts/world-v1/model/jepa.pt')
    bundle=torch.load('artifacts/world-v4/contact/contact-head.pt',map_location='cpu',weights_only=False)
    head=ContactHead(bundle['latent_size'],bundle['hidden_size']);head.load_state_dict(bundle['state_dict']);head.eval()
    x,y,seeds,audit=corrected.build_inputs(episodes,labels,base,head)
    return x,y,seeds,audit


def run_offline():
    cfg=json.loads(CONFIG.read_text());ROOT.mkdir(parents=True,exist_ok=True)
    protocol_path=ROOT/'protocol.json'
    if protocol_path.exists():raise FileExistsError('V13 offline protocol exists.')
    files=[CONFIG,Path('configs/world_contact_v4.json'),Path('src/tinyvla/world_hybrid_v13_experiment.py'),
      Path('src/tinyvla/world_hybrid_ranker_v7.py'),Path('src/tinyvla/world_contact_targeted.py'),
      Path('src/tinyvla/world_contact_data.py'),Path('src/tinyvla/world_contact_rank.py'),Path('src/tinyvla/world_contact_reranker.py'),
      Path('src/tinyvla/world_contact_train.py'),Path('src/tinyvla/world_model.py'),Path('src/tinyvla/world_data.py'),
      Path('src/tinyvla/sim.py'),Path('src/tinyvla/evaluate.py'),Path('src/tinyvla/language.py'),Path('uv.lock'),
      Path('artifacts/model/best.pt'),Path('artifacts/world-v1/model/jepa.pt'),Path('artifacts/world-v4/contact/contact-head.pt'),
      V5/'protocol.json',V5/'branches_train.npz',V5/'branches_validation.npz',V5/'ranker.pt']
    protocol={'version':1,'created_unix':int(time.time()),'task_label':'Complete pick, lift, and place success in PyBullet; no physical arm.',
      'train_scenes':cfg['ranking']['train_scenes'],'validation_scenes':cfg['ranking']['validation_scenes'],
      'test_scenes':cfg['ranking']['test_scenes'],'test_seed_range':[cfg['ranking']['test_seed'],cfg['ranking']['test_seed']+cfg['ranking']['test_scenes']-1],
      'candidate_count':25,'candidate_branch':'Open at the selected grasp point, close, lift, then execute the simulator teacher transport path.',
      'features':'Corrected V7 contact channel, JEPA held score, candidate and predicted target geometry; tipped simulator labels omitted.',
      'source_sha256':{str(path):digest(path) for path in files}}
    protocol_path.write_text(json.dumps(protocol,indent=2)+'\n')
    # Build corrected inputs and full-task outcomes for V5 train/validation only.
    splits={}
    for split in ('train','validation'):
        episodes=V5/'episodes'/split;labels,audit=collect_full_success_labels(episodes,split,cfg)
        x,y,seeds,feature_audit=build_split_features(episodes,labels)
        np.savez_compressed(ROOT/f'branches_{split}.npz',features=x,labels=y,scene_seed=seeds)
        splits[split]={'scenes':int(len(np.unique(seeds))),'success_branches':int(y.sum()),'branch_audit':audit,'feature_audit':feature_audit}
    # Fresh held-out scenes are collected only after the ranker and validation selection are frozen.
    base_config=json.loads(Path('configs/world_contact_v4.json').read_text())
    spec=cfg['ranking'];base_config['data']['test_episodes']=spec['test_scenes'];base_config['data']['test_seed']=spec['test_seed']
    base_config['data']['tip_probability']=.8;base_config['data']['drop_probability']=.45
    base_config['data']['high_noise_probability']=.65;base_config['data']['high_noise_std']=.02;base_config['data']['low_noise_std']=.008
    targeted.ROOT=ROOT/'episodes';targeted.collect_targeted(base_config,'test')
    labels,audit=collect_full_success_labels(ROOT/'episodes/test','test',cfg)
    x,y,seeds,feature_audit=build_split_features(ROOT/'episodes/test',labels)
    np.savez_compressed(ROOT/'branches_test.npz',features=x,labels=y,scene_seed=seeds)
    splits['test']={'scenes':int(len(np.unique(seeds))),'success_branches':int(y.sum()),'branch_audit':audit,'feature_audit':feature_audit}
    settings=cfg['training'];torch.manual_seed(settings['seed']);np.random.seed(settings['seed']);torch.use_deterministic_algorithms(True);torch.set_num_threads(1)
    with np.load(ROOT/'branches_train.npz') as d:xt,yt=d['features'],d['labels']
    with np.load(ROOT/'branches_validation.npz') as d:xv,yv,sv=d['features'],d['labels'],d['scene_seed']
    mean=xt.mean(0);std=xt.std(0).clip(1e-5);tx=torch.from_numpy((xt-mean)/std);ty=torch.from_numpy(yt);vx=torch.from_numpy((xv-mean)/std)
    model=Ranker(xt.shape[1],settings['hidden_size']);pos=max(float(yt.sum()),1);neg=max(float(len(yt)-yt.sum()),1)
    weight=torch.tensor(min(settings['positive_weight_max'],max(1.,neg/pos)));opt=torch.optim.AdamW(model.parameters(),lr=settings['learning_rate'])
    best_ap=-1;best=None;best_step=0;log=[]
    for step in range(1,settings['steps']+1):
        model.train();score=model(tx);loss=nn.functional.binary_cross_entropy_with_logits(score,ty,pos_weight=weight)
        opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
        if step==1 or step%10==0 or step==settings['steps']:
            model.eval()
            with torch.no_grad():scores=model(vx).numpy()
            metric=scene_metrics(scores,yv,sv);log.append({'step':step,'loss':float(loss.detach()),'validation':metric})
            if np.isfinite(metric['macro_average_precision']) and metric['macro_average_precision']>best_ap:
                best_ap=metric['macro_average_precision'];best_step=step;best={k:v.detach().clone() for k,v in model.state_dict().items()}
    if best is None:raise ValueError('V13 validation split contains no positive task-success branches.')
    model.load_state_dict(best);model.eval()
    torch.save({'state_dict':model.state_dict(),'mean':mean,'std':std,'feature_names':corrected.FEATURE_NAMES,
      'hidden_size':settings['hidden_size'],'selected_step':best_step,'protocol_sha256':digest(protocol_path)},ROOT/'ranker.pt')
    (ROOT/'training.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in log))
    with np.load(ROOT/'branches_test.npz') as d:xe,ye,se=d['features'],d['labels'],d['scene_seed']
    with torch.no_grad():scores=model(torch.from_numpy((xe-mean)/std)).numpy()
    baseline=-np.tile(np.arange(25),len(ye)//25)
    comparison={'task_success_ranker':scene_metrics(scores,ye,se),'unchanged_grasp':scene_metrics(baseline,ye,se)}
    wins=ties=losses=0;scene_rows=[]
    for seed in np.unique(se):
        m=se==seed;label=ye[m];selected=bool(label[int(scores[m].argmax())]);original=bool(label[0])
        if selected and not original:wins+=1
        elif selected==original:ties+=1
        else:losses+=1
        scene_rows.append({'seed':int(seed),'ranker_success':selected,'unchanged_success':original,
                           'ranker_candidate':int(scores[m].argmax()),'positive_candidates':int(label.sum())})
    result={'protocol_sha256':digest(protocol_path),'ranker_sha256':digest(ROOT/'ranker.pt'),'split_summary':
      {k:{'scenes':v['scenes'],'success_branches':v['success_branches']} for k,v in splits.items()},
      'selected_step':best_step,'parameters':sum(p.numel() for p in model.parameters()),
      'test_metrics':comparison,'paired_vs_unchanged':{'wins':wins,'ties':ties,'losses':losses},'scene_rows':scene_rows,
      'interpretation':'Offline complete-task branches. Labels include the scripted transport path and simulator score; this does not establish closed-loop task success.'}
    (ROOT/'manifest.json').write_text(json.dumps({'protocol_sha256':digest(protocol_path),
      'datasets':{f'branches_{name}.npz':digest(ROOT/f'branches_{name}.npz') for name in splits}},indent=2)+'\n')
    (ROOT/'offline_results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='scene_rows'},indent=2),flush=True)
    return result


def run_controller():
    cfg=json.loads(CONFIG.read_text());off=json.loads((ROOT/'offline_results.json').read_text())
    protocol_path=ROOT/'controller_protocol.json'
    if protocol_path.exists():raise FileExistsError('V13 controller protocol exists.')
    pair=off['paired_vs_unchanged']
    if pair['wins']<=pair['losses']:
        result={'controller_test_run':False,'reason':'Task-success ranker did not beat unchanged grasp on fresh offline branches.',
                'offline_paired':pair}
        (ROOT/'controller_decision.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True);return result
    files=[CONFIG,Path('configs/project.json'),Path('uv.lock'),Path('src/tinyvla/world_hybrid_v13_experiment.py'),
      Path('src/tinyvla/world_hybrid_corrected.py'),Path('src/tinyvla/world_hybrid_ranker_v7.py'),Path('src/tinyvla/world_evaluate.py'),
      Path('src/tinyvla/world_planner.py'),Path('src/tinyvla/controller.py'),Path('src/tinyvla/world_model.py'),
      Path('src/tinyvla/world_contact_rank.py'),Path('src/tinyvla/world_contact_reranker.py'),Path('src/tinyvla/world_contact_train.py'),
      Path('src/tinyvla/world_data.py'),Path('src/tinyvla/evaluate.py'),Path('src/tinyvla/sim.py'),Path('src/tinyvla/language.py'),
      Path('artifacts/model/best.pt'),Path('artifacts/world-v1/model/jepa.pt'),Path('artifacts/world-v4/contact/contact-head.pt'),
      ROOT/'ranker.pt',ROOT/'protocol.json',ROOT/'offline_results.json']
    controller_cfg=json.loads(json.dumps(cfg));controller_cfg['evaluation']['final_seed']=cfg['ranking']['controller_seed']
    controller_protocol={'version':1,'created_unix':int(time.time()),'seed_start':controller_cfg['evaluation']['final_seed'],
      'episodes_per_condition':controller_cfg['evaluation']['episodes'],'conditions':controller_cfg['evaluation']['suites'],
      'offline_model_sha256':digest(ROOT/'ranker.pt'),'source_sha256':{str(p):digest(p) for p in files}}
    protocol_path.write_text(json.dumps(controller_protocol,indent=2)+'\n')
    world_evaluate.HybridGraspPlanner=CorrectedHybridGraspPlanner
    result=world_evaluate.benchmark(controller_cfg,ROOT/'controller','artifacts/model/best.pt')
    rows=[]
    for suite in controller_cfg['evaluation']['suites']:
        folder=ROOT/'controller/evaluation'/suite
        b={r['seed']:r for r in map(json.loads,(folder/'feedback.jsonl').read_text().splitlines())}
        h={r['seed']:r for r in map(json.loads,(folder/'hybrid.jsonl').read_text().splitlines())}
        for seed in b:
            rows.append({'suite':suite,'seed':seed,'feedback_success':b[seed]['success'],'hybrid_success':h[seed]['success'],
              'feedback_status':b[seed]['status'],'hybrid_status':h[seed]['status'],'feedback_safe_stop':b[seed]['safe_stop'],
              'hybrid_safe_stop':h[seed]['safe_stop'],'planning':h[seed]['planning']})
    task=[r for r in rows if r['suite']!='missing'];wins=sum(r['hybrid_success'] and not r['feedback_success'] for r in task)
    losses=sum(r['feedback_success'] and not r['hybrid_success'] for r in task);ties=len(task)-wins-losses
    summary={'task_trials':len(task),'feedback_successes':sum(r['feedback_success'] for r in task),'hybrid_successes':sum(r['hybrid_success'] for r in task),
      'wins':wins,'losses':losses,'ties':ties,'missing_trials':sum(r['suite']=='missing' for r in rows),
      'feedback_missing_safe_stops':sum(r['suite']=='missing' and r['feedback_safe_stop'] for r in rows),
      'hybrid_missing_safe_stops':sum(r['suite']=='missing' and r['hybrid_safe_stop'] for r in rows),
      'protocol_sha256':digest(protocol_path),'evaluation_seconds':result['seconds'],'rows':rows}
    (ROOT/'paired_results.json').write_text(json.dumps(summary,indent=2)+'\n')
    (ROOT/'RESULTS.md').write_text('\n'.join(['# V13: rank grasps by complete simulated task success','',
      'All work used PyBullet. No physical robot was used.','',
      'The candidate ranker was trained with a full pick, lift, and place outcome for each grasp candidate. It selected a candidate by predicted task success rather than predicted contact alone. Fresh offline branches were used before a matched closed-loop controller test. The live controller still retained camera checks, finger checks, bounded offsets, and retry limits.','',
      f"Closed-loop results: feedback {summary['feedback_successes']}/{summary['task_trials']}; V13 hybrid {summary['hybrid_successes']}/{summary['task_trials']}. Paired outcomes were {wins} wins, {losses} losses, and {ties} ties.",'',
      f"Missing-object safe stops: feedback {summary['feedback_missing_safe_stops']}/{summary['missing_trials']}; V13 hybrid {summary['hybrid_missing_safe_stops']}/{summary['missing_trials']}.",'',
      'See the paired results and per-seed videos for the full record.'])+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='rows'},indent=2),flush=True);return summary


def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['offline','controller','all']);cmd=parser.parse_args().command
    offline=None
    if cmd in ('offline','all'):offline=run_offline()
    if cmd in ('controller','all'):run_controller()


if __name__=='__main__':main()

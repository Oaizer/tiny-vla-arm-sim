"""Train a visual tipped-object gate and route to the adapted policy only when needed."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from . import world_evaluate
from .evaluate import load_policy
from .language import parse
from .sim import ArmSim
from .world_data import VisualFeatures, tip_object

ROOT=Path('artifacts/world-v10-mlp/visual-router')
CONFIG=Path('configs/world_v10_router.json')
BASE=Path('artifacts/model/best.pt')
ADAPTED=Path('artifacts/world-v8/tipped-adaptation/model/adapted.pt')


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def detector_model(feature_size):
    return nn.Sequential(nn.Linear(feature_size,32),nn.GELU(),nn.Linear(32,1))


def features_for(policy, images, states, calibrations):
    extractor=VisualFeatures(policy); out=[]
    for image,state,calibration in zip(images,states,calibrations):
        out.append(extractor({'rgb':image,'state':state,'calibration':calibration}))
    return np.stack(out).astype(np.float32)


def fresh_data(policy, start, count, tipped):
    sim=ArmSim(128); task=parse('lift the red cylinder'); rows=[]
    try:
        for i in range(count):
            obs=sim.reset(start+i,omissions=False)
            if tipped:tip_object(sim,task);obs=sim.observation()
            rows.append(VisualFeatures(policy)(obs))
    finally:sim.close()
    return np.asarray(rows,np.float32)


def load_scene_arrays(path):
    with np.load(path) as d:
        return d['images'],d['states'],d['calibration']


def train_detector():
    cfg=json.loads(CONFIG.read_text());ROOT.mkdir(parents=True,exist_ok=True)
    detector_path=ROOT/'tip_detector.pt'
    if detector_path.exists():raise FileExistsError('V10 detector exists; keep it immutable.')
    detector_protocol=ROOT/'detector_protocol.json'
    if detector_protocol.exists():raise FileExistsError('V10 detector protocol exists.')
    frozen_inputs=[CONFIG,Path('src/tinyvla/world_v10_router.py'),Path('src/tinyvla/world_data.py'),
      Path('src/tinyvla/model.py'),Path('src/tinyvla/sim.py'),Path('src/tinyvla/language.py'),Path('uv.lock'),BASE,
      Path('artifacts/world-v8/tipped-adaptation/data/train.npz'),Path('artifacts/world-v8/tipped-adaptation/data/validation.npz'),
      Path('artifacts/data/train.npz'),Path('artifacts/data/validation.npz')]
    detector_protocol.write_text(json.dumps({'version':1,'created_unix':int(time.time()),
      'classifier':'537-to-32-to-1 MLP on frozen visual features','train_positive_scenes':240,
      'train_negative_scenes':240,'validation_positive_scenes':60,'validation_negative_scenes':60,
      'fresh_test_scenes_per_class':cfg['fresh_detector_scenes_per_class'],
      'source_sha256':{str(path):digest(path) for path in frozen_inputs}},indent=2)+'\n')
    policy,_=load_policy(BASE);torch.set_num_threads(1);torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed'])
    train_pos=features_for(policy,*load_scene_arrays('artifacts/world-v8/tipped-adaptation/data/train.npz'))
    val_pos=features_for(policy,*load_scene_arrays('artifacts/world-v8/tipped-adaptation/data/validation.npz'))
    neg_x,neg_s,neg_c=load_scene_arrays('artifacts/data/train.npz');
    train_neg=features_for(policy,neg_x[:len(train_pos)],neg_s[:len(train_pos)],neg_c[:len(train_pos)])
    val_x,val_s,val_c=load_scene_arrays('artifacts/data/validation.npz')
    val_neg=features_for(policy,val_x[:len(val_pos)],val_s[:len(val_pos)],val_c[:len(val_pos)])
    x_train=np.concatenate([train_pos,train_neg]);y_train=np.r_[np.ones(len(train_pos)),np.zeros(len(train_neg))].astype(np.float32)
    x_val=np.concatenate([val_pos,val_neg]);y_val=np.r_[np.ones(len(val_pos)),np.zeros(len(val_neg))].astype(np.float32)
    mean=x_train.mean(0);std=x_train.std(0).clip(1e-4)
    tx=torch.from_numpy((x_train-mean)/std);ty=torch.from_numpy(y_train)
    vx=torch.from_numpy((x_val-mean)/std)
    model=detector_model(x_train.shape[1]);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    history=[];best=None;best_loss=float('inf')
    for step in range(1,801):
        model.train();logits=model(tx).squeeze(-1);loss=nn.functional.binary_cross_entropy_with_logits(logits,ty)
        opt.zero_grad();loss.backward();opt.step()
        if step%20==0 or step==1:
            model.eval()
            with torch.no_grad(): scores=model(vx).squeeze(-1).sigmoid().numpy()
            with torch.no_grad():
                logits=model(vx).squeeze(-1)
                val_loss=float(nn.functional.binary_cross_entropy_with_logits(logits,torch.from_numpy(y_val)))
            history.append({'step':step,'loss':float(loss.detach()),'validation_loss':val_loss})
            if val_loss<best_loss:best_loss=val_loss;best={k:v.detach().clone() for k,v in model.state_dict().items()};best_step=step
    model.load_state_dict(best);model.eval()
    with torch.no_grad(): val_scores=model(vx).squeeze(-1).sigmoid().numpy()
    # Select the lowest threshold with <=5% false positives, then favor recall.
    thresholds=np.unique(np.r_[0.,val_scores,1.])
    valid=[float(t) for t in thresholds if np.mean(val_scores[y_val==0]>=t)<=cfg['max_validation_false_positive_rate']]
    threshold=max(valid,key=lambda t:float(np.mean(val_scores[y_val==1]>=t)))
    fresh_pos=fresh_data(policy,cfg['fresh_detector_seed'],cfg['fresh_detector_scenes_per_class'],True)
    fresh_neg=fresh_data(policy,cfg['fresh_detector_seed']+1000,cfg['fresh_detector_scenes_per_class'],False)
    x_test=np.concatenate([fresh_pos,fresh_neg]);y_test=np.r_[np.ones(len(fresh_pos)),np.zeros(len(fresh_neg))].astype(np.float32)
    with torch.no_grad():test_scores=model(torch.from_numpy((x_test-mean)/std)).squeeze(-1).sigmoid().numpy()
    model_bundle={'state_dict':model.state_dict(),'mean':mean,'std':std,'threshold':threshold,'feature_size':x_train.shape[1],
                  'selected_step':best_step,'seed':cfg['seed']}
    torch.save(model_bundle,detector_path)
    np.savez_compressed(ROOT/'detector_test.npz',scores=test_scores,labels=y_test,
                        seeds=np.r_[np.arange(cfg['fresh_detector_seed'],cfg['fresh_detector_seed']+len(fresh_pos)),
                                    np.arange(cfg['fresh_detector_seed']+1000,cfg['fresh_detector_seed']+1000+len(fresh_neg))])
    (ROOT/'detector_training.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in history))
    test={'threshold':threshold,'validation_positive_recall':float(np.mean(val_scores[y_val==1]>=threshold)),
      'validation_false_positive_rate':float(np.mean(val_scores[y_val==0]>=threshold)),
      'test_tipped_recall':float(np.mean(test_scores[y_test==1]>=threshold)),
      'test_nominal_false_positive_rate':float(np.mean(test_scores[y_test==0]>=threshold)),
      'test_scenes_per_class':len(fresh_pos),'test_seed_ranges':[[cfg['fresh_detector_seed'],cfg['fresh_detector_seed']+len(fresh_pos)-1],
        [cfg['fresh_detector_seed']+1000,cfg['fresh_detector_seed']+1000+len(fresh_neg)-1]],'selected_step':best_step}
    (ROOT/'detector_results.json').write_text(json.dumps(test,indent=2)+'\n')
    return test


class RoutedPolicy:
    def __init__(self,base,adapted,detector_path):
        self.base=base.eval();self.adapted=adapted.eval();self.vision=self.base.vision
        bundle=torch.load(detector_path,map_location='cpu',weights_only=False)
        self.detector=detector_model(bundle['feature_size']);self.detector.load_state_dict(bundle['state_dict']);self.detector.eval()
        self.mean=torch.as_tensor(bundle['mean']);self.std=torch.as_tensor(bundle['std']);self.threshold=float(bundle['threshold'])
        self.last_route=[]
    def eval(self):self.base.eval();self.adapted.eval();return self
    def __call__(self,rgb,tokens,state,calibration):
        with torch.inference_mode():
            observation={'rgb':(rgb[0].permute(1,2,0).cpu().numpy()*255).clip(0,255).astype(np.uint8),
                         'state':state[0].cpu().numpy(),'calibration':calibration[0].cpu().numpy()}
            feature=VisualFeatures(self.base)(observation)
            probability=float(self.detector((torch.from_numpy(feature)-self.mean)/self.std).sigmoid().item())
        chosen=self.adapted if probability>=self.threshold else self.base
        self.last_route.append({'tipped_probability':probability,'adapted_policy_used':probability>=self.threshold})
        return chosen(rgb,tokens,state,calibration)


def evaluate():
    cfg=json.loads(CONFIG.read_text());ROOT.mkdir(parents=True,exist_ok=True)
    if (ROOT/'protocol.json').exists():raise FileExistsError('V10 task protocol exists.')
    detector_results=json.loads((ROOT/'detector_results.json').read_text())
    protocol_path=ROOT/'protocol.json'
    sources=[CONFIG,Path('src/tinyvla/world_v10_router.py'),Path('src/tinyvla/model.py'),Path('src/tinyvla/data.py'),
      Path('src/tinyvla/evaluate.py'),Path('src/tinyvla/world_evaluate.py'),Path('src/tinyvla/controller.py'),
      Path('src/tinyvla/world_data.py'),Path('src/tinyvla/sim.py'),Path('src/tinyvla/language.py'),Path('uv.lock'),
      BASE,ADAPTED,ROOT/'tip_detector.pt',ROOT/'detector_results.json',ROOT/'detector_protocol.json']
    protocol={'version':1,'created_unix':int(time.time()),'hypothesis':'A visual tipped-object gate can use the domain-adapted TinyVLA only for fallen objects and retain the base policy elsewhere.',
      'detector_threshold':detector_results['threshold'],'detector_fresh_test':detector_results,
      'evaluation_seed':cfg['controller_seed'],'episodes_per_condition':cfg['episodes_per_condition'],'conditions':cfg['conditions'],
      'source_sha256':{str(p):digest(p) for p in sources}}
    protocol_path.write_text(json.dumps(protocol,indent=2)+'\n')
    task_cfg={'task_instruction':'put the red cylinder on the yellow pad','model_root':'artifacts/world-v1/model','planner':{},
      'evaluation':{'episodes':cfg['episodes_per_condition'],'final_seed':cfg['controller_seed'],'inference_threads':1,'suites':cfg['conditions']},'final_models':[]}
    baseline_root=ROOT/'evaluation'/'baseline';routed_root=ROOT/'evaluation'/'routed'
    baseline_root.mkdir(parents=True,exist_ok=True);routed_root.mkdir(parents=True,exist_ok=True)
    world_evaluate.benchmark(task_cfg,baseline_root,BASE)
    base,_=load_policy(BASE);adapted,_=load_policy(ADAPTED);routed=RoutedPolicy(base,adapted,ROOT/'tip_detector.pt')
    original_loader=world_evaluate.load_policy
    world_evaluate.load_policy=lambda _path:(routed,{})
    try:world_evaluate.benchmark(task_cfg,routed_root,BASE)
    finally:world_evaluate.load_policy=original_loader
    rows=[]
    for suite in cfg['conditions']:
        bfile=baseline_root/'evaluation'/suite/'feedback.jsonl';rfile=routed_root/'evaluation'/suite/'feedback.jsonl'
        b={r['seed']:r for r in map(json.loads,bfile.read_text().splitlines())};r={x['seed']:x for x in map(json.loads,rfile.read_text().splitlines())}
        if b.keys()!=r.keys():raise ValueError(f'Unmatched V10 seeds in {suite}.')
        for seed in sorted(b):
            row={'suite':suite,'seed':seed,'baseline_success':b[seed]['success'],'routed_success':r[seed]['success'],
              'baseline_status':b[seed]['status'],'routed_status':r[seed]['status'],
              'baseline_safe_stop':b[seed]['safe_stop'],'routed_safe_stop':r[seed]['safe_stop']}
            rows.append(row)
    task=[x for x in rows if x['suite']!='missing'];wins=sum(x['routed_success'] and not x['baseline_success'] for x in task)
    losses=sum(x['baseline_success'] and not x['routed_success'] for x in task)
    summary={'task_trials':len(task),'baseline_successes':sum(x['baseline_success'] for x in task),
      'routed_successes':sum(x['routed_success'] for x in task),'wins':wins,'losses':losses,'ties':len(task)-wins-losses,
      'missing_trials':sum(x['suite']=='missing' for x in rows),
      'baseline_missing_safe_stops':sum(x['suite']=='missing' and x['baseline_safe_stop'] for x in rows),
      'routed_missing_safe_stops':sum(x['suite']=='missing' and x['routed_safe_stop'] for x in rows),
      'detector':detector_results,'rows':rows,'protocol_sha256':digest(protocol_path)}
    (ROOT/'paired_results.json').write_text(json.dumps(summary,indent=2)+'\n')
    (ROOT/'RESULTS.md').write_text('\n'.join(['# V10: route the adapted policy only for fallen objects','',
      'This experiment used only simulated images and actions.','',
      f"A tiny visual gate trained on frozen vision features routed to the adapted policy at probability {detector_results['threshold']:.3f}. On fresh detector images, recall for tipped objects was {detector_results['test_tipped_recall']:.1%}, and the false-positive rate on nominal scenes was {detector_results['test_nominal_false_positive_rate']:.1%}.",'',
      '| Policy | Task successes | Missing-object safe stops |','|---|---:|---:|',
      f"| Base | {summary['baseline_successes']}/{summary['task_trials']} | {summary['baseline_missing_safe_stops']}/{summary['missing_trials']} |",
      f"| Routed | {summary['routed_successes']}/{summary['task_trials']} | {summary['routed_missing_safe_stops']}/{summary['missing_trials']} |",'',
      f"Paired outcomes: {wins} wins, {losses} losses, {summary['ties']} ties for routing.",'',
      'The gate does not read the simulator condition or object pose. It predicts from the image and arm observation. See `paired_results.json` for every seed.'])+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='rows'},indent=2),flush=True)


def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['train','evaluate','all']);cmd=parser.parse_args().command
    if cmd in ('train','all'):train_detector()
    if cmd in ('evaluate','all'):evaluate()


if __name__=='__main__':main()

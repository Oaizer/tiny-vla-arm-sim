"""Train a small crop classifier to route tipped scenes to the adapted policy."""
from __future__ import annotations

import hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from . import world_evaluate
from .evaluate import load_policy
from .language import OBJECTS,parse,tokenize
from .sim import ArmSim
from .world_data import tip_object

ROOT=Path('artifacts/world-v10-crop/router')
CONFIG=Path('configs/world_v10_crop_router.json')
BASE=Path('artifacts/model/best.pt')
ADAPTED=Path('artifacts/world-v8/tipped-adaptation/model/adapted.pt')
INSTRUCTION='put the red cylinder on the yellow pad'
TOKEN=torch.from_numpy(tokenize(INSTRUCTION))[None]
CROP=32


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_detector():
    return nn.Sequential(nn.Conv2d(3,8,3,padding=1),nn.GELU(),nn.MaxPool2d(2),
      nn.Conv2d(8,16,3,padding=1),nn.GELU(),nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(16,1))


def crop_at(image,uv):
    h,w=image.shape[-2:];cx=int(round(float(uv[0])*(w-1)));cy=int(round(float(uv[1])*(h-1)))
    x0=min(max(0,cx-CROP//2),w-CROP);y0=min(max(0,cy-CROP//2),h-CROP)
    return image[:,y0:y0+CROP,x0:x0+CROP]


@torch.inference_mode()
def crops_from_arrays(policy,images,states,calibrations):
    result=[];source=OBJECTS.index('red cylinder')
    for start in range(0,len(images),32):
        rgb=torch.from_numpy(images[start:start+32]).permute(0,3,1,2).float()/255
        state=torch.from_numpy(states[start:start+32]).float();calib=torch.from_numpy(calibrations[start:start+32]).float()
        tokens=TOKEN.expand(len(rgb),-1)
        _,info=policy.features(rgb,tokens,state,calib)
        for image,uv in zip(rgb,info['pixels'][:,source]):result.append(crop_at(image,uv))
    return torch.stack(result)


def load_arrays(path):
    with np.load(path) as data:return data['images'],data['states'],data['calibration']


def fresh_crops(policy,start,count,tipped):
    sim=ArmSim(128);task=parse('lift the red cylinder');images=[];states=[];calibrations=[]
    try:
        for seed in range(start,start+count):
            obs=sim.reset(seed)
            if tipped:tip_object(sim,task);obs=sim.observation()
            images.append(obs['rgb']);states.append(obs['state']);calibrations.append(obs['calibration'])
    finally:sim.close()
    return crops_from_arrays(policy,np.asarray(images),np.asarray(states),np.asarray(calibrations))


def train_detector():
    cfg=json.loads(CONFIG.read_text());ROOT.mkdir(parents=True,exist_ok=True)
    if (ROOT/'tip_detector.pt').exists() or (ROOT/'detector_protocol.json').exists():raise FileExistsError('Crop detector output exists.')
    files=[CONFIG,Path('src/tinyvla/world_v10_crop_router.py'),Path('src/tinyvla/model.py'),Path('src/tinyvla/sim.py'),
      Path('src/tinyvla/world_data.py'),Path('src/tinyvla/language.py'),Path('uv.lock'),BASE,ADAPTED,
      Path('artifacts/world-v8/tipped-adaptation/data/train.npz'),Path('artifacts/world-v8/tipped-adaptation/data/validation.npz'),
      Path('artifacts/data/train.npz'),Path('artifacts/data/validation.npz')]
    (ROOT/'detector_protocol.json').write_text(json.dumps({'version':1,'created_unix':int(time.time()),
      'classifier':'32x32 RGB crop around policy red-cylinder attention; two-convolution network',
      'train_scenes_per_class':240,'validation_scenes_per_class':60,
      'fresh_test_scenes_per_class':cfg['fresh_detector_scenes_per_class'],
      'source_sha256':{str(p):digest(p) for p in files}},indent=2)+'\n')
    base,_=load_policy(BASE);torch.set_num_threads(1);torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed'])
    pos=crops_from_arrays(base,*load_arrays('artifacts/world-v8/tipped-adaptation/data/train.npz'))
    valpos=crops_from_arrays(base,*load_arrays('artifacts/world-v8/tipped-adaptation/data/validation.npz'))
    images,states,calibrations=load_arrays('artifacts/data/train.npz')
    neg=crops_from_arrays(base,images[:len(pos)],states[:len(pos)],calibrations[:len(pos)])
    images,states,calibrations=load_arrays('artifacts/data/validation.npz')
    valneg=crops_from_arrays(base,images[:len(valpos)],states[:len(valpos)],calibrations[:len(valpos)])
    x=torch.cat([pos,neg]);y=torch.cat([torch.ones(len(pos)),torch.zeros(len(neg))])
    vx=torch.cat([valpos,valneg]);vy=torch.cat([torch.ones(len(valpos)),torch.zeros(len(valneg))])
    detector=make_detector();opt=torch.optim.AdamW(detector.parameters(),lr=.001,weight_decay=.01)
    best_loss=float('inf');best=None;log=[]
    for step in range(1,1201):
        detector.train();idx=torch.randint(len(x),(64,));xb=x[idx].clone();yb=y[idx]
        # Mild photometric and translation jitter improves tolerance to camera noise.
        xb=(xb*torch.empty(len(xb),1,1,1).uniform_(.88,1.12)).clamp(0,1)
        logits=detector(xb).squeeze(-1);loss=F.binary_cross_entropy_with_logits(logits,yb)
        opt.zero_grad();loss.backward();opt.step()
        if step==1 or step%20==0 or step==1200:
            detector.eval()
            with torch.no_grad():
                val_logits=detector(vx).squeeze(-1);val_loss=float(F.binary_cross_entropy_with_logits(val_logits,vy))
            row={'step':step,'loss':float(loss.detach()),'validation_loss':val_loss};log.append(row)
            if val_loss<best_loss:best_loss=val_loss;best={k:v.detach().clone() for k,v in detector.state_dict().items()};best_step=step
    detector.load_state_dict(best);detector.eval()
    with torch.no_grad():val_scores=detector(vx).squeeze(-1).sigmoid().numpy();labels=vy.numpy()
    thresholds=np.unique(np.r_[0.,val_scores,1.])
    allowed=[float(t) for t in thresholds if np.mean(val_scores[labels==0]>=t)<=cfg['max_validation_false_positive_rate']]
    threshold=max(allowed,key=lambda t:float(np.mean(val_scores[labels==1]>=t)))
    testpos=fresh_crops(base,cfg['fresh_detector_seed'],cfg['fresh_detector_scenes_per_class'],True)
    testneg=fresh_crops(base,cfg['fresh_detector_seed']+1000,cfg['fresh_detector_scenes_per_class'],False)
    testx=torch.cat([testpos,testneg]);testy=np.r_[np.ones(len(testpos)),np.zeros(len(testneg))]
    with torch.no_grad():scores=detector(testx).squeeze(-1).sigmoid().numpy()
    torch.save({'state_dict':detector.state_dict(),'threshold':threshold,'selected_step':best_step,'crop_size':CROP},ROOT/'tip_detector.pt')
    (ROOT/'training.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in log))
    result={'threshold':threshold,'validation_recall':float(np.mean(val_scores[labels==1]>=threshold)),
      'validation_false_positive_rate':float(np.mean(val_scores[labels==0]>=threshold)),
      'fresh_tipped_recall':float(np.mean(scores[testy==1]>=threshold)),
      'fresh_nominal_false_positive_rate':float(np.mean(scores[testy==0]>=threshold)),
      'scenes_per_class':cfg['fresh_detector_scenes_per_class'],'selected_step':best_step}
    (ROOT/'detector_results.json').write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(ROOT/'detector_test.npz',scores=scores,labels=testy)
    print(json.dumps(result,indent=2),flush=True);return result


class RoutedPolicy:
    def __init__(self,base,adapted,path):
        self.base=base.eval();self.adapted=adapted.eval();self.vision=self.base.vision
        bundle=torch.load(path,map_location='cpu',weights_only=False);self.detector=make_detector();self.detector.load_state_dict(bundle['state_dict']);self.detector.eval()
        self.threshold=float(bundle['threshold']);self.last_route=[]
    def eval(self):self.base.eval();self.adapted.eval();return self
    def __call__(self,rgb,tokens,state,calibration):
        with torch.inference_mode():
            _,info=self.base.features(rgb,tokens,state,calibration)
            crop=crop_at(rgb[0],info['pixels'][0,OBJECTS.index('red cylinder')]).unsqueeze(0)
            prob=float(self.detector(crop).sigmoid().item());use=prob>=self.threshold
            self.last_route.append({'tipped_probability':prob,'adapted_policy_used':use})
        return (self.adapted if use else self.base)(rgb,tokens,state,calibration)


def evaluate():
    cfg=json.loads(CONFIG.read_text());ROOT.mkdir(parents=True,exist_ok=True)
    if (ROOT/'protocol.json').exists():raise FileExistsError('Crop routing test protocol exists.')
    detector_result=json.loads((ROOT/'detector_results.json').read_text());protocol_path=ROOT/'protocol.json'
    files=[CONFIG,Path('src/tinyvla/world_v10_crop_router.py'),Path('src/tinyvla/model.py'),Path('src/tinyvla/evaluate.py'),
      Path('src/tinyvla/world_evaluate.py'),Path('src/tinyvla/controller.py'),Path('src/tinyvla/world_data.py'),
      Path('src/tinyvla/sim.py'),Path('src/tinyvla/language.py'),Path('uv.lock'),BASE,ADAPTED,ROOT/'tip_detector.pt',ROOT/'detector_results.json']
    protocol={'version':1,'created_unix':int(time.time()),'detector_fresh_test':detector_result,
      'controller_seed':cfg['controller_seed'],'episodes_per_condition':cfg['episodes_per_condition'],'conditions':cfg['conditions'],
      'hypothesis':'A visual crop gate can preserve the base model on upright objects and route fallen objects to the simulator-adapted policy.',
      'source_sha256':{str(p):digest(p) for p in files}}
    protocol_path.write_text(json.dumps(protocol,indent=2)+'\n')
    taskcfg={'task_instruction':INSTRUCTION,'model_root':'artifacts/world-v1/model','planner':{},
      'evaluation':{'episodes':cfg['episodes_per_condition'],'final_seed':cfg['controller_seed'],'inference_threads':1,'suites':cfg['conditions']},'final_models':[]}
    base_root=ROOT/'evaluation'/'base';route_root=ROOT/'evaluation'/'routed';base_root.mkdir(parents=True,exist_ok=True);route_root.mkdir(parents=True,exist_ok=True)
    world_evaluate.benchmark(taskcfg,base_root,BASE)
    base,_=load_policy(BASE);adapted,_=load_policy(ADAPTED);route=RoutedPolicy(base,adapted,ROOT/'tip_detector.pt')
    old_loader=world_evaluate.load_policy;world_evaluate.load_policy=lambda _:(route,{})
    try:world_evaluate.benchmark(taskcfg,route_root,BASE)
    finally:world_evaluate.load_policy=old_loader
    rows=[]
    for suite in cfg['conditions']:
        def read(root):return {r['seed']:r for r in map(json.loads,(root/'evaluation'/suite/'feedback.jsonl').read_text().splitlines())}
        b=read(base_root);r=read(route_root)
        if b.keys()!=r.keys():raise ValueError(f'Unmatched seeds in {suite}.')
        for seed in b:rows.append({'suite':suite,'seed':seed,'base_success':b[seed]['success'],'routed_success':r[seed]['success'],
          'base_status':b[seed]['status'],'routed_status':r[seed]['status'],'base_safe_stop':b[seed]['safe_stop'],'routed_safe_stop':r[seed]['safe_stop']})
    task=[x for x in rows if x['suite']!='missing'];wins=sum(x['routed_success'] and not x['base_success'] for x in task);losses=sum(x['base_success'] and not x['routed_success'] for x in task)
    summary={'task_trials':len(task),'base_successes':sum(x['base_success'] for x in task),'routed_successes':sum(x['routed_success'] for x in task),
      'wins':wins,'losses':losses,'ties':len(task)-wins-losses,'missing_trials':sum(x['suite']=='missing' for x in rows),
      'base_missing_safe_stops':sum(x['suite']=='missing' and x['base_safe_stop'] for x in rows),
      'routed_missing_safe_stops':sum(x['suite']=='missing' and x['routed_safe_stop'] for x in rows),
      'detector':detector_result,'rows':rows,'protocol_sha256':digest(protocol_path)}
    (ROOT/'paired_results.json').write_text(json.dumps(summary,indent=2)+'\n')
    (ROOT/'RESULTS.md').write_text('\n'.join(['# V10: crop-based visual routing','',
      'All trials used simulated scenes and a simulated arm.','',
      f"On fresh classifier scenes, the crop detector found {detector_result['fresh_tipped_recall']:.1%} of fallen objects with {detector_result['fresh_nominal_false_positive_rate']:.1%} false positives on upright scenes.",'',
      '| Policy | Task successes | Missing-object safe stops |','|---|---:|---:|',
      f"| Base | {summary['base_successes']}/{summary['task_trials']} | {summary['base_missing_safe_stops']}/{summary['missing_trials']} |",
      f"| Routed | {summary['routed_successes']}/{summary['task_trials']} | {summary['routed_missing_safe_stops']}/{summary['missing_trials']} |",'',
      f"Paired outcomes: {wins} wins, {losses} losses, {summary['ties']} ties for routing.",'',
      'Every seed outcome and controller video is stored in `evaluation/` and `paired_results.json`.'])+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='rows'},indent=2),flush=True)


def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['train','evaluate','all']);cmd=parser.parse_args().command
    if cmd in ('train','all'):train_detector()
    if cmd in ('evaluate','all'):evaluate()


if __name__=='__main__':main()

"""Fine tune TinyVLA on tipped-object simulation scenes and compare frozen policies."""
from __future__ import annotations

import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from .data import load
from .evaluate import load_policy
from .language import OBJECTS, PADS, TASKS, commands, parse, tokenize
from .sim import ArmSim
from .world_data import tip_object
from . import world_evaluate

ROOT = Path('artifacts/world-v8/tipped-adaptation')
CONFIG = Path('configs/world_v8_adaptation.json')
BASE_POLICY = Path('artifacts/model/best.pt')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def collect(split, count, seed_start, size, examples, seed):
    output = ROOT / 'data' / f'{split}.npz'
    if output.exists():
        raise FileExistsError(f'Preserve existing adaptation data: {output}')
    sim = ArmSim(size)
    texts = commands(); tokens = np.stack([tokenize(text) for text in texts]); tasks = [parse(text) for text in texts]
    choices = [i for i, task in enumerate(tasks) if task.object == 'red cylinder']
    image_rows=[]; states=[]; calibrations=[]; pixels=[]; present=[]; scene=[]; instructions=[]; actions=[]; valid=[]
    try:
        for offset in range(count):
            episode_seed = seed_start + offset
            obs = sim.reset(episode_seed, omissions=True)
            tip_object(sim, parse('lift the red cylinder'))
            obs = sim.observation()
            _, uv, exists = sim.labels()
            image_rows.append(obs['rgb']); states.append(obs['state']); calibrations.append(obs['calibration'])
            pixels.append(uv); present.append(exists)
            rng = np.random.default_rng(seed + offset)
            chosen = rng.choice(choices, examples, replace=False)
            for command in chosen:
                task = tasks[int(command)]
                available = task.object in sim.objects and (task.kind == 'lift' or task.pad in sim.pads)
                scene.append(offset); instructions.append(int(command)); valid.append(available)
                action = sim.teacher(task) if available else np.zeros((8, 4), np.float32)
                # Use the settled fallen pose for the first pick waypoints.
                if available:
                    xyz, _ = __import__('pybullet').getBasePositionAndOrientation(sim.objects[task.object])
                    action[0, :2] = xyz[:2]; action[1, :2] = xyz[:2]; action[2, :2] = xyz[:2]
                    action[:3, 2] = [.20, .035, .035]
                actions.append(action)
            if (offset + 1) % 50 == 0:
                print(json.dumps({'split': split, 'scenes': offset+1, 'total': count}), flush=True)
    finally:
        sim.close()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, images=np.asarray(image_rows), states=np.asarray(states),
        calibration=np.asarray(calibrations), pixels=np.asarray(pixels), present=np.asarray(present),
        scene=np.asarray(scene), instruction=np.asarray(instructions), actions=np.asarray(actions),
        valid=np.asarray(valid, np.float32), tokens=tokens, texts=texts,
        source=np.asarray([OBJECTS.index(task.object) for task in tasks]),
        pad=np.asarray([PADS.index(task.pad) if task.pad else 0 for task in tasks]),
        task=np.asarray([TASKS.index(task.kind) for task in tasks]),
        seeds=np.arange(seed_start, seed_start+count))
    return output


def indexed_batch(data, ids):
    scene = data['scene'][ids]; instruction = data['instruction'][ids]
    return {name: data[name][scene] for name in ('images','states','calibration','pixels','present')} | {
        name: data[name][instruction] for name in ('tokens','source','pad','task')} | {
        name: data[name][ids] for name in ('actions','valid')}


def validate(model, old, tipped):
    model.eval()
    rows=[]
    with torch.inference_mode():
        for name, data in [('original', old), ('tipped', tipped)]:
            total=0;xy=0.;correct=0;count=0
            for start in range(0, len(data['scene']), 64):
                ids=torch.arange(start,min(start+64,len(data['scene'])))
                batch=indexed_batch(data, ids); pred,info=model(batch['images'],batch['tokens'],batch['states'],batch['calibration'])
                valid=batch['valid']; scale=model.scale
                xy += (((pred[:,:,:2]-batch['actions'][:,:,:2]).square().mean((1,2)))*valid).sum().item()
                count += valid.sum().item(); correct += (info['source_logits'].argmax(-1)==batch['source']).sum().item(); total += len(ids)
            rows.append({'split':name,'xy_rmse_mm':float(np.sqrt(xy/max(count,1))*1000),'source_accuracy':correct/max(total,1)})
    return rows


def train():
    cfg=json.loads(CONFIG.read_text()); out=ROOT/'model'; out.mkdir(parents=True,exist_ok=True)
    if (out/'adapted.pt').exists(): raise FileExistsError('V8 model exists. Keep it immutable.')
    data_protocol=ROOT/'data_protocol.json'
    if data_protocol.exists(): raise FileExistsError('V8 data protocol already exists.')
    sources=[CONFIG,Path('src/tinyvla/world_v8_adaptation.py'),Path('src/tinyvla/model.py'),
             Path('src/tinyvla/data.py'),Path('src/tinyvla/sim.py'),Path('src/tinyvla/language.py'),
             Path('src/tinyvla/world_data.py'),Path('uv.lock'),BASE_POLICY,
             Path('artifacts/data/train.npz'),Path('artifacts/data/validation.npz')]
    protocol={'version':1,'created_unix':int(time.time()),'seed':cfg['seed'],
      'train_seed_range':[cfg['train_seed'],cfg['train_seed']+cfg['train_scenes']-1],
      'validation_seed_range':[cfg['validation_seed'],cfg['validation_seed']+cfg['validation_scenes']-1],
      'task_scope':'Red-cylinder instructions; fallen red cylinder; simulation-generated labels.',
      'fine_tuning':{'steps':cfg['fine_tune_steps'],'batch_size':cfg['batch_size'],
                     'learning_rate':cfg['learning_rate'],'target_mix':cfg['target_mix']},
      'source_sha256':{str(path):digest(path) for path in sources}}
    data_protocol.write_text(json.dumps(protocol,indent=2)+'\n')
    train_path=collect('train',cfg['train_scenes'],cfg['train_seed'],cfg['image_size'],cfg['examples_per_scene'],cfg['seed'])
    validation_path=collect('validation',cfg['validation_scenes'],cfg['validation_seed'],cfg['image_size'],cfg['examples_per_scene'],cfg['seed']+1)
    old_train=load('artifacts/data/train.npz'); old_val=load('artifacts/data/validation.npz')
    target_train=load(train_path); target_val=load(validation_path)
    model, checkpoint=load_policy(BASE_POLICY); torch.manual_seed(cfg['seed']); np.random.seed(cfg['seed']); random.seed(cfg['seed'])
    torch.use_deterministic_algorithms(True); torch.set_num_threads(1)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=1e-5)
    best=None; best_score=float('inf'); best_step=0; log=[]; started=time.time()
    for step in range(1,cfg['fine_tune_steps']+1):
        model.train(); n=cfg['batch_size']; nt=int(round(n*cfg['target_mix'])); no=n-nt
        ids_t=torch.randint(len(target_train['scene']),(nt,)); ids_o=torch.randint(len(old_train['scene']),(no,))
        bt=indexed_batch(target_train,ids_t); bo=indexed_batch(old_train,ids_o)
        batch={key:torch.cat([bt[key],bo[key]],0) for key in bt}
        order=torch.randperm(n); batch={key:value[order] for key,value in batch.items()}
        loss,_,_=model.loss(batch); optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5); optimizer.step()
        if step==1 or step%20==0 or step==cfg['fine_tune_steps']:
            target_metrics=validate(model,old_val,target_val)
            tipped=target_metrics[1]; original=target_metrics[0]
            score=tipped['xy_rmse_mm'] + 100*(1-tipped['source_accuracy']) + max(0,original['xy_rmse_mm']-12)*.25
            row={'step':step,'loss':float(loss.detach()),'validation':target_metrics,'selection_score':score,'seconds':time.time()-started}
            log.append(row); print(json.dumps(row),flush=True)
            if score<best_score:
                best_score=score; best_step=step; best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(best); model.eval()
    artifact={'state_dict':model.state_dict(),'architecture':checkpoint.get('architecture'),'step':best_step,
              'parameters':sum(p.numel() for p in model.parameters()),'config':checkpoint['config'],
              'seed_ranges':checkpoint['seed_ranges'],'adaptation':cfg,
              'adaptation_seed_ranges':{'train':[cfg['train_seed'],cfg['train_seed']+cfg['train_scenes']-1],
                                        'validation':[cfg['validation_seed'],cfg['validation_seed']+cfg['validation_scenes']-1]},
              'validation':validate(model,old_val,target_val),'base_sha256':digest(BASE_POLICY),
              'data_protocol_sha256':digest(data_protocol)}
    torch.save(artifact,out/'adapted.pt')
    (ROOT/'training.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in log))
    return artifact


def evaluate():
    cfg=json.loads(CONFIG.read_text()); run_root=ROOT/'evaluation'; run_root.mkdir(parents=True,exist_ok=True)
    protocol_path=ROOT/'protocol.json'
    if protocol_path.exists(): raise FileExistsError('V8 evaluation protocol exists.')
    protocol={'version':1,'created_unix':int(time.time()),'hypothesis':'Simulation fine-tuning on fallen red-cylinder demonstrations improves visual localization and task completion on tipped red-cylinder trials.',
      'train_seed_range':[cfg['train_seed'],cfg['train_seed']+cfg['train_scenes']-1],
      'validation_seed_range':[cfg['validation_seed'],cfg['validation_seed']+cfg['validation_scenes']-1],
      'evaluation_seed':cfg['evaluation_seed'],'episodes_per_condition':cfg['episodes_per_condition'],'conditions':cfg['conditions'],
      'source_sha256':{str(path):digest(path) for path in [CONFIG,Path('src/tinyvla/world_v8_adaptation.py'),Path('src/tinyvla/model.py'),Path('src/tinyvla/data.py'),Path('src/tinyvla/sim.py'),Path('src/tinyvla/controller.py'),Path('src/tinyvla/evaluate.py'),Path('src/tinyvla/world_evaluate.py'),Path('src/tinyvla/language.py'),Path('uv.lock'),BASE_POLICY,Path('artifacts/data/train.npz'),Path('artifacts/data/validation.npz'),ROOT/'model/adapted.pt',ROOT/'data/train.npz',ROOT/'data/validation.npz']}}
    protocol_path.write_text(json.dumps(protocol,indent=2)+'\n')
    config={'task_instruction':'put the red cylinder on the yellow pad','model_root':'artifacts/world-v1/model','planner':{},
            'evaluation':{'episodes':cfg['episodes_per_condition'],'final_seed':cfg['evaluation_seed'],'inference_threads':1,'suites':cfg['conditions']},
            'final_models':[]}
    base_root=run_root/'baseline'; candidate_root=run_root/'adapted'
    base_root.mkdir(parents=True, exist_ok=True); candidate_root.mkdir(parents=True, exist_ok=True)
    world_evaluate.benchmark(config,base_root,BASE_POLICY)
    world_evaluate.benchmark(config,candidate_root,ROOT/'model/adapted.pt')
    rows=[]
    for suite_index,suite in enumerate(cfg['conditions']):
        def read(root):
            return {r['seed']:r for r in map(json.loads,(root/'evaluation'/suite/'feedback.jsonl').read_text().splitlines())}
        base=read(base_root); cand=read(candidate_root)
        if base.keys()!=cand.keys(): raise ValueError('V8 paired seeds do not match.')
        for seed in sorted(base):
            rows.append({'suite':suite,'seed':seed,'baseline_success':base[seed]['success'],'adapted_success':cand[seed]['success'],
                         'baseline_status':base[seed]['status'],'adapted_status':cand[seed]['status'],
                         'baseline_safe_stop':base[seed]['safe_stop'],'adapted_safe_stop':cand[seed]['safe_stop']})
    task=[r for r in rows if r['suite']!='missing']; wins=sum(r['adapted_success'] and not r['baseline_success'] for r in task)
    losses=sum(r['baseline_success'] and not r['adapted_success'] for r in task)
    summary={'task_trials':len(task),'baseline_successes':sum(r['baseline_success'] for r in task),
      'adapted_successes':sum(r['adapted_success'] for r in task),'wins':wins,'losses':losses,'ties':len(task)-wins-losses,
      'missing_trials':sum(r['suite']=='missing' for r in rows),
      'baseline_missing_safe_stops':sum(r['suite']=='missing' and r['baseline_safe_stop'] for r in rows),
      'adapted_missing_safe_stops':sum(r['suite']=='missing' and r['adapted_safe_stop'] for r in rows),
      'rows':rows,'protocol_sha256':digest(protocol_path),'adapted_model_sha256':digest(ROOT/'model/adapted.pt')}
    (run_root/'paired_results.json').write_text(json.dumps(summary,indent=2)+'\n')
    (run_root/'RESULTS.md').write_text('\n'.join(['# V8: adapt the vision and action model to fallen objects','',
      'All trials used PyBullet. No physical arm was used.','',
      f"The adapted model was fine-tuned on {cfg['train_scenes']} simulated scenes with a fallen red cylinder. It used a 50:50 mix of those examples and the original training examples. The base model and adapted model received identical fresh evaluation seeds.",'',
      '| Model | Task successes | Missing-object safe stops |','|---|---:|---:|',
      f"| Base | {summary['baseline_successes']}/{summary['task_trials']} | {summary['baseline_missing_safe_stops']}/{summary['missing_trials']} |",
      f"| Adapted | {summary['adapted_successes']}/{summary['task_trials']} | {summary['adapted_missing_safe_stops']}/{summary['missing_trials']} |",'',
      f"Paired outcomes: {wins} wins, {losses} losses, {summary['ties']} ties for adaptation.",'',
      'The complete per-seed outcomes, videos, hashes, and training record are stored beside this report. The test covers one object, one placement instruction, and a small number of simulated scenes.'])+'\n')
    return summary


def main():
    import argparse
    parser=argparse.ArgumentParser(); parser.add_argument('command',choices=['train','evaluate','all']); command=parser.parse_args().command
    if command in ('train','all'): train()
    if command in ('evaluate','all'): evaluate()


if __name__=='__main__': main()

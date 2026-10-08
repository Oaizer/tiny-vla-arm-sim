"""Deterministic data generation with scene-level train/validation split."""
import json,time
from pathlib import Path
import numpy as np
import torch
from .sim import ArmSim
from .language import OBJECTS,PADS,TASKS,commands,parse,tokenize

def generate(output,count,start_seed,size=128,examples=3):
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    sim=ArmSim(size);texts=commands();tokens=np.stack([tokenize(t) for t in texts]);tasks=[parse(t) for t in texts]
    images=[];states=[];calibrations=[];pixels=[];present=[];scene_indices=[];instructions=[];actions=[];valid=[]
    begin=time.time()
    try:
        for i in range(count):
            seed=start_seed+i;obs=sim.reset(seed,omissions=True)
            points,uv,exists=sim.labels();images.append(obs['rgb']);states.append(obs['state']);calibrations.append(obs['calibration']);pixels.append(uv);present.append(exists)
            rng=np.random.default_rng(seed+800000)
            for command in rng.choice(len(texts),examples,replace=False):
                task=tasks[command];available=task.object in sim.objects and (task.kind=='lift' or task.pad in sim.pads)
                scene_indices.append(i);instructions.append(command);valid.append(available)
                actions.append(sim.teacher(task) if available else np.zeros((8,4),np.float32))
            if (i+1)%200==0:print(json.dumps({'generation_scenes':i+1,'elapsed_s':round(time.time()-begin,1)}),flush=True)
    finally:sim.close()
    np.savez_compressed(output,images=np.array(images),states=np.array(states),calibration=np.array(calibrations),
        pixels=np.array(pixels),present=np.array(present),scene=np.array(scene_indices),instruction=np.array(instructions),
        actions=np.array(actions),valid=np.array(valid,np.float32),tokens=tokens,texts=texts,
        source=np.array([OBJECTS.index(t.object) for t in tasks]),pad=np.array([PADS.index(t.pad) if t.pad else 0 for t in tasks]),
        task=np.array([TASKS.index(t.kind) for t in tasks]),seeds=np.arange(start_seed,start_seed+count))
    return {'scenes':count,'examples':len(actions),'seed_start':start_seed,'seed_end':start_seed+count-1,'seconds':time.time()-begin}

def load(path):
    raw=np.load(path)
    result={name:torch.from_numpy(raw[name]) for name in ['states','calibration','pixels','present','scene','instruction','actions','valid','tokens','source','pad','task']}
    result['images']=torch.from_numpy(raw['images']).permute(0,3,1,2).float()/255
    result['seeds']=raw['seeds'];return result

def batch(data,indices):
    scene=data['scene'][indices];instruction=data['instruction'][indices]
    result={k:data[k][scene] for k in ['images','states','calibration','pixels','present']}
    result.update({k:data[k][instruction] for k in ['tokens','source','pad','task']})
    result.update({k:data[k][indices] for k in ['actions','valid']});return result

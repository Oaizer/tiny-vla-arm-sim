"""Fit the linear action head on training features only; select using validation."""
import json
from pathlib import Path
import torch
from model import TinyVLA
from train import load,validate

torch.set_num_threads(4)
path=Path('runs/tiny/best.pt');ckpt=torch.load(path,weights_only=False)
model=TinyVLA();model.load_state_dict(ckpt['state_dict']);model.eval()
tr=load('data/train.npz');va=load('data/val.npz');xs=[];ys=[]
before=validate(model,va)
with torch.inference_mode():
    for start in range(0,len(tr['images'])*6,96):
        idx=torch.arange(start,min(start+96,len(tr['images'])*6));scene=idx//6;ins=idx%6
        _,_,coords=model(tr['images'][scene],tr['tokens'][ins],tr['states'][scene])
        x=torch.cat([coords.flatten(1),(tr['states'][scene]-model.offset)/model.scale,torch.ones(len(idx),1)],1)
        y=((tr['actions'][scene,ins]-model.offset)/model.scale).flatten(1)
        xs.append(x.double());ys.append(y.double())
    x=torch.cat(xs);y=torch.cat(ys)
    ridge=torch.eye(x.shape[1],dtype=torch.float64)*.001
    beta=torch.linalg.solve(x.T@x+ridge,x.T@y).float()
    model.action_head.weight.copy_(beta[:-1].T);model.action_head.bias.copy_(beta[-1])
after=validate(model,va)
accepted=after['normalized_mse']<before['normalized_mse']
report={'before':before,'after':after,'accepted':accepted,'training_pairs':len(x),'ridge':.001}
print(json.dumps(report,indent=2),flush=True)
path.with_name('refit.json').write_text(json.dumps(report,indent=2))
if accepted:
    torch.save(ckpt,path.with_name('best_before_refit.pt'))
    ckpt['state_dict']=model.state_dict();ckpt['metrics']=after;ckpt['linear_head_refit']=report
    torch.save(ckpt,path)

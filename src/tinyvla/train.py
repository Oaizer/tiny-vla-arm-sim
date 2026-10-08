"""CPU training, validation-only selection, and training-only linear head refit."""
import json,time,random
from pathlib import Path
import numpy as np
import torch
from .model import Policy
from .data import load,batch

@torch.inference_mode()
def validate(model,data):
    total=0;xy_sum=0.;mse_sum=0.;valid_count=0.;source_ok=0;task_ok=0;pad_ok=0;place_count=0;presence_correct=0;presence_total=0
    for start in range(0,len(data['scene']),64):
        b=batch(data,torch.arange(start,min(start+64,len(data['scene']))));pred,info=model(b['images'],b['tokens'],b['states'],b['calibration'])
        valid=b['valid'];valid_count+=valid.sum().item()
        mse_sum+=((((pred-b['actions'])/model.scale).square().mean((1,2)))*valid).sum().item()
        xy_sum+=((pred[:,:,:2]-b['actions'][:,:,:2]).square().mean((1,2))*valid).sum().item()
        total+=len(pred);source_ok+=(info['source_logits'].argmax(-1)==b['source']).sum().item()
        task_ok+=(info['task_logits'].argmax(-1)==b['task']).sum().item();place=b['task']==0;place_count+=place.sum().item()
        pad_ok+=((info['pad_logits'].argmax(-1)==b['pad'])&place).sum().item()
        presence_correct+=((info['presence']>.65)==(b['present']>.5)).sum().item();presence_total+=b['present'].numel()
    return {'normalized_action_mse':mse_sum/max(valid_count,1),'xy_rmse_mm':float(np.sqrt(xy_sum/max(valid_count,1))*1000),
            'source_accuracy':source_ok/total,'task_accuracy':task_ok/total,'pad_accuracy':pad_ok/max(place_count,1),
            'presence_accuracy':presence_correct/presence_total,'valid_examples':valid_count}

@torch.inference_mode()
def refit(model,data):
    xs=[[],[]];ys=[[],[]]
    for start in range(0,len(data['scene']),64):
        b=batch(data,torch.arange(start,min(start+64,len(data['scene']))))
        features,_=model.features(b['images'],b['tokens'],b['states'],b['calibration'])
        # Teacher emits absolute waypoints independent of the initial robot pose.
        # Constrain pose coefficients to zero when fitting that absolute-action head.
        design=torch.cat([features[:,:4],torch.ones(len(features),1)],1).double()
        target=((b['actions']-model.offset)/model.scale).flatten(1).double()
        for task in range(2):
            mask=(b['task']==task)&(b['valid']>.5);xs[task].append(design[mask]);ys[task].append(target[mask])
    for task in range(2):
        x=torch.cat(xs[task]);y=torch.cat(ys[task]);ridge=torch.eye(x.shape[1],dtype=torch.float64)*.001
        beta=torch.linalg.solve(x.T@x+ridge,x.T@y).float();section=slice(task*32,(task+1)*32)
        model.action_head.weight[section].zero_();model.action_head.weight[section,:4].copy_(beta[:-1].T)
        model.action_head.bias[section].copy_(beta[-1])

def train(config,artifact_root):
    root=Path(artifact_root);run=root/'model';run.mkdir(parents=True,exist_ok=True)
    cfg=config['training'];seed=cfg['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.set_num_threads(cfg['threads']);torch.use_deterministic_algorithms(True)
    tr=load(root/'data/train.npz');va=load(root/'data/validation.npz')
    if set(tr['seeds'])&set(va['seeds']):raise ValueError('Train/validation seed overlap')
    model=Policy();params=sum(p.numel() for p in model.parameters())
    if params>config['model_goal']['max_parameters']:raise ValueError('Model exceeds parameter budget')
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=1e-5)
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,cfg['steps'],eta_min=.0001)
    best=float('inf');start=time.time();checkpoint=run/'best.pt'
    print(json.dumps({'training_parameters':params,'examples':len(tr['scene'])}),flush=True)
    with (run/'training.jsonl').open('w') as log:
        for step in range(1,cfg['steps']+1):
            model.train();b=batch(tr,torch.randint(len(tr['scene']),(cfg['batch_size'],)))
            # Additional mild brightness noise beyond simulator randomization.
            b['images']=(b['images']*torch.empty(len(b['images']),1,1,1).uniform_(.9,1.1)).clamp(0,1)
            loss,_,_=model.loss(b);optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5)
            optimizer.step();schedule.step()
            if step==1 or step%200==0 or step==cfg['steps']:
                model.eval();metrics=validate(model,va);row={'step':step,'loss':float(loss.detach()),**metrics,'seconds':round(time.time()-start,1)}
                print(json.dumps(row),flush=True);log.write(json.dumps(row)+'\n');log.flush()
                criterion=metrics['normalized_action_mse']+1-metrics['source_accuracy']+1-metrics['task_accuracy']
                if criterion<best:
                    best=criterion;torch.save({'state_dict':model.state_dict(),'architecture':'TinyVLA-v2',
                        'step':step,'metrics':metrics,'parameters':params,'config':config,'seed_ranges':{
                            'train':[int(tr['seeds'].min()),int(tr['seeds'].max())],
                            'validation':[int(va['seeds'].min()),int(va['seeds'].max())]}},checkpoint)
    ckpt=torch.load(checkpoint,weights_only=False);model.load_state_dict(ckpt['state_dict']);model.eval()
    before=validate(model,va);refit(model,tr);after=validate(model,va)
    accepted=after['normalized_action_mse']<before['normalized_action_mse']
    if accepted:ckpt['state_dict']=model.state_dict();ckpt['metrics']=after
    ckpt['head_refit']={'before':before,'after':after,'accepted':accepted};ckpt['training_seconds']=time.time()-start
    torch.save(ckpt,checkpoint);(run/'head_refit.json').write_text(json.dumps(ckpt['head_refit'],indent=2))
    print(json.dumps({'completed_training':str(checkpoint),'head_refit':ckpt['head_refit'],'seconds':ckpt['training_seconds']}),flush=True)
    return ckpt

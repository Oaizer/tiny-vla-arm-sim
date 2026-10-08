"""Train and select checkpoints on a disjoint validation set; no test seeds used."""
import argparse,json,time,math
from pathlib import Path
import numpy as np
import torch
from model import TinyVLA

def load(path):
    d=np.load(path)
    return {'images':torch.from_numpy(d['images']).permute(0,3,1,2).float()/255,
            'states':torch.from_numpy(d['states']).float(),
            'actions':torch.from_numpy(d['actions']).float(),
            'pixels':torch.from_numpy(d['pixels']).float(),
            'tokens':torch.from_numpy(d['tokens']).long()}

@torch.inference_mode()
def validate(model,data):
    errs=[];xy=[];coords=[]
    for start in range(0,len(data['images'])*6,96):
        idx=torch.arange(start,min(start+96,len(data['images'])*6));scene=idx//6;ins=idx%6
        pred,_,co=model(data['images'][scene],data['tokens'][ins],data['states'][scene])
        truth=data['actions'][scene,ins]
        errs.append(((pred-truth)/model.scale).square().mean().item())
        xy.append((pred[:,:,:2]-truth[:,:,:2]).square().mean().item())
        coords.append((co-(data['pixels'][scene,ins]*2-1)).square().mean().item())
    return {'normalized_mse':float(np.mean(errs)),'xy_rmse_mm':float(np.sqrt(np.mean(xy))*1000),
            'coord_rmse':float(np.sqrt(np.mean(coords)))}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--train',default='data/train.npz')
    ap.add_argument('--val',default='data/val.npz');ap.add_argument('--steps',type=int,default=3000)
    ap.add_argument('--batch',type=int,default=48);ap.add_argument('--threads',type=int,default=4)
    ap.add_argument('--output',default='runs/tiny');ap.add_argument('--resume')
    args=ap.parse_args();torch.set_num_threads(args.threads);torch.manual_seed(42)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    tr=load(args.train);va=load(args.val);model=TinyVLA()
    if args.resume:model.load_state_dict(torch.load(args.resume,weights_only=False)['state_dict'])
    opt=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=1e-5)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,args.steps,eta_min=.0001)
    best=float('inf');t=time.time();params=sum(p.numel() for p in model.parameters())
    print(json.dumps({'parameters':params,'train_scenes':len(tr['images']),'val_scenes':len(va['images'])}),flush=True)
    with (out/'train.jsonl').open('w') as log:
        for step in range(1,args.steps+1):
            model.train();scene=torch.randint(len(tr['images']),(args.batch,));ins=torch.randint(6,(args.batch,))
            image=tr['images'][scene]
            # Brightness augmentation does not reveal task/object metadata.
            image=(image*torch.empty(args.batch,1,1,1).uniform_(.9,1.1)).clamp(0,1)
            loss,_=model.loss(image,tr['tokens'][ins],tr['states'][scene],tr['actions'][scene,ins],tr['pixels'][scene,ins])
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();scheduler.step()
            if step%100==0 or step==1 or step==args.steps:
                model.eval();metrics=validate(model,va)
                row={'step':step,'loss':loss.item(),**metrics,'seconds':round(time.time()-t,1)}
                print(json.dumps(row),flush=True);log.write(json.dumps(row)+'\n');log.flush()
                if metrics['normalized_mse']<best:
                    best=metrics['normalized_mse'];torch.save({'state_dict':model.state_dict(),
                        'step':step,'metrics':metrics,'parameters':params,'training_seed':42,
                        'architecture':'TinyVLA-v1','args':vars(args)},out/'best.pt')
        torch.save({'state_dict':model.state_dict(),'step':args.steps,'parameters':params},out/'last.pt')
    print(f'Saved best model: {out}/best.pt',flush=True)

if __name__=='__main__':main()

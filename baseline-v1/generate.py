"""Generate RGB/language/action demonstrations using simulator-only teacher labels."""
import argparse,json,time
from pathlib import Path
import numpy as np
from simulator import PandaSim,COLORS,PADS,tokenize

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--scenes',type=int,default=1200)
    ap.add_argument('--start-seed',type=int,default=0);ap.add_argument('--output',default='data/train.npz')
    args=ap.parse_args();sim=PandaSim();images=[];states=[];actions=[];pixels=[]
    instructions=[f'put the {c} block on the {p} pad' for c in COLORS for p in PADS]
    tokens=np.stack([tokenize(s) for s in instructions]);t=time.time()
    for i in range(args.scenes):
        image,state=sim.reset(args.start_seed+i)
        images.append(image);states.append(state)
        aa=[];pp=[]
        for c in COLORS:
            for p in PADS:
                src,dst=sim.truth(c,p)
                aa.append(sim.teacher(c,p));pp.append([sim.pixel(src+[0,0,.02]),sim.pixel(dst)])
        actions.append(aa);pixels.append(pp)
        if (i+1)%100==0:print(json.dumps({'scenes':i+1,'seconds':round(time.time()-t,1)}),flush=True)
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out,images=np.array(images),states=np.array(states),
        actions=np.array(actions),pixels=np.array(pixels),tokens=tokens,
        seeds=np.arange(args.start_seed,args.start_seed+args.scenes),instructions=instructions)
    sim.close();print(f'Saved {out}: {args.scenes} scenes, {args.scenes*6} instruction/action pairs',flush=True)

if __name__=='__main__':main()

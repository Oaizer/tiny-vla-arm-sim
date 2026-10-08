"""Evaluate real physics outcomes on held-out scene seeds; save each result/video."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from simulator import PandaSim,COLORS,PADS,tokenize
from model import TinyVLA

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',default='runs/tiny/best.pt')
    ap.add_argument('--episodes',type=int,default=100);ap.add_argument('--start-seed',type=int,default=20000)
    ap.add_argument('--output',default='evaluation');ap.add_argument('--videos',type=int,default=6)
    ap.add_argument('--teacher',action='store_true');ap.add_argument('--ablate',choices=['none','blank_image','wrong_language'],default='none')
    args=ap.parse_args();torch.set_num_threads(4)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    model=TinyVLA();ckpt=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    model.load_state_dict(ckpt['state_dict']);model.eval();sim=PandaSim();rows=[];latencies=[]
    with (out/'episodes.jsonl').open('w') as log:
        for i in range(args.episodes):
            seed=args.start_seed+i;color=list(COLORS)[(i//2)%3];pad=list(PADS)[i%2]
            instruction=f'put the {color} block on the {pad} pad'
            image,state=sim.reset(seed);frames=[]
            if args.ablate=='blank_image':image[:]=0
            text=instruction
            if args.ablate=='wrong_language':text=f'put the {list(COLORS)[((i//2)+1)%3]} block on the {pad} pad'
            ts=time.perf_counter()
            with torch.inference_mode():
                action,_,_=model(torch.from_numpy(image).permute(2,0,1)[None].float()/255,
                                torch.from_numpy(tokenize(text))[None],torch.from_numpy(state)[None])
            ms=(time.perf_counter()-ts)*1000;latencies.append(ms)
            predicted=action[0].numpy()
            if args.teacher:predicted=sim.teacher(color,pad)
            sim.execute(predicted,frames if i<args.videos else None)
            success,distance=sim.score(color,pad)
            row={'seed':seed,'instruction':instruction,'policy_instruction':text,'success':success,
                 'distance_m':distance,'max_lift_height_m':sim.max_heights[color],
                 'inference_ms':ms,'waypoints':predicted.tolist()}
            rows.append(row);log.write(json.dumps(row)+'\n');log.flush()
            print(json.dumps({k:v for k,v in row.items() if k!='waypoints'}),flush=True)
            if frames:
                import imageio.v2 as imageio
                imageio.mimsave(out/f'{i:03d}_{color}_{pad}_{"success" if success else "failure"}.mp4',frames,fps=20)
    n=len(rows);k=sum(r['success'] for r in rows);z=1.96;ph=k/n
    center=(ph+z*z/(2*n))/(1+z*z/n);half=z*np.sqrt(ph*(1-ph)/n+z*z/(4*n*n))/(1+z*z/n)
    result={'episodes':n,'successes':k,'success_rate':k/n,'wilson_95':[center-half,center+half],
        'latency_p50_ms':float(np.median(latencies)),'latency_p95_ms':float(np.percentile(latencies,95)),
        'parameters':sum(p.numel() for p in model.parameters()),'checkpoint_bytes':Path(args.checkpoint).stat().st_size,
        'checkpoint':args.checkpoint,'start_seed':args.start_seed,'ablation':args.ablate,'teacher':args.teacher,
        'by_instruction':{s:{'episodes':sum(r['instruction']==s for r in rows),
            'successes':sum(r['success'] and r['instruction']==s for r in rows)} for s in sorted(set(r['instruction'] for r in rows))}}
    (out/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True);sim.close()

if __name__=='__main__':main()

"""Run one instruction through the trained policy and record the simulated arm."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
import imageio.v2 as imageio
from PIL import Image,ImageDraw
from simulator import PandaSim,COLORS,PADS,tokenize
from model import TinyVLA

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--instruction',default='put the red block on the yellow pad')
    ap.add_argument('--seed',type=int,default=30000);ap.add_argument('--checkpoint',default='runs/tiny/best.pt')
    ap.add_argument('--output',default='demo.mp4');args=ap.parse_args()
    valid={f'put the {c} block on the {p} pad':(c,p) for c in COLORS for p in PADS}
    if args.instruction not in valid:ap.error('Use: put the [red|green|blue] block on the [yellow|purple] pad')
    torch.set_num_threads(4);model=TinyVLA();model.load_state_dict(torch.load(args.checkpoint,weights_only=False)['state_dict']);model.eval()
    sim=PandaSim();rgb,state=sim.reset(args.seed)
    with torch.inference_mode():
        action,_,_=model(torch.from_numpy(rgb).permute(2,0,1)[None].float()/255,
             torch.from_numpy(tokenize(args.instruction))[None],torch.from_numpy(state)[None])
    frames=[sim.image(384,overview=True)]*12
    sim.execute(action[0].numpy(),frames)
    frames += [sim.image(384,overview=True)]*20
    color,pad=valid[args.instruction];ok,distance=sim.score(color,pad)
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    with imageio.get_writer(output,fps=20) as writer:
        for rgb in frames:
            canvas=Image.new('RGB',(384,432),(20,25,35));canvas.paste(Image.fromarray(rgb),(0,48))
            draw=ImageDraw.Draw(canvas);draw.text((10,8),'TINY VLA | learned policy | Panda simulation',fill='white')
            draw.text((10,27),args.instruction,fill=(180,225,255));writer.append_data(np.array(canvas))
    report={'instruction':args.instruction,'seed':args.seed,'success':ok,'distance_m':distance,
            'max_lift_height_m':sim.max_heights[color],'video':str(output)}
    output.with_suffix('.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2));sim.close()

if __name__=='__main__':main()

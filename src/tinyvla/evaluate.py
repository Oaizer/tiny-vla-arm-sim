"""Paired benchmark suites. Perturbation harness is separate from policy inputs."""
import hashlib,json,time
from pathlib import Path
import numpy as np
import torch
import imageio.v2 as imageio
from PIL import Image,ImageDraw
from .sim import ArmSim
from .model import Policy
from .controller import Agent
from .language import OBJECTS,PADS,parse

def load_policy(checkpoint):
    ckpt=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if ckpt.get('architecture')!='TinyVLA-v2':raise ValueError('Checkpoint architecture mismatch')
    model=Policy();model.load_state_dict(ckpt['state_dict']);model.eval();return model,ckpt

def instruction_for(index,heldout=False):
    obj=OBJECTS[(index//4)%4];target=index%4
    if target==3:return f'pick up {obj}' if heldout else f'lift the {obj}'
    return f'move {obj} onto {PADS[target]} target' if heldout else f'put the {obj} on the {PADS[target]} pad'

def annotate_video(path,frames,instruction,label):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if not frames:return
    with imageio.get_writer(path,fps=20) as writer:
        for rgb in frames:
            canvas=Image.new('RGB',(384,432),(20,25,35));canvas.paste(Image.fromarray(rgb),(0,48));draw=ImageDraw.Draw(canvas)
            draw.text((8,7),label,fill='white');draw.text((8,26),instruction,fill=(175,225,255));writer.append_data(np.asarray(canvas))

def interval(k,n):
    if not n:return [0.,1.]
    z=1.96;q=k/n;center=(q+z*z/(2*n))/(1+z*z/n);half=z*np.sqrt(q*(1-q)/n+z*z/(4*n*n))/(1+z*z/n)
    return [max(0.,center-half),min(1.,center+half)]

def benchmark(config,run_root,suites=None,episodes=None,videos=None,seed_start=None,output_name='evaluation'):
    root=Path(run_root);checkpoint=root/'model/best.pt';model,ckpt=load_policy(checkpoint)
    torch.set_num_threads(config['training']['threads']);settings=config['evaluation']
    suites=suites or settings['suites'];episodes=episodes or settings['episodes_per_suite']
    videos=settings['videos_per_suite'] if videos is None else videos
    seed_start=settings['seed_start'] if seed_start is None else seed_start
    used=set(range(ckpt['seed_ranges']['train'][0],ckpt['seed_ranges']['train'][1]+1))|set(range(ckpt['seed_ranges']['validation'][0],ckpt['seed_ranges']['validation'][1]+1))
    output=root/output_name;output.mkdir(parents=True,exist_ok=True);summaries=[];begin=time.time()
    digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    for suite_index,suite in enumerate(suites):
        suite_root=output/suite;suite_root.mkdir(parents=True,exist_ok=True)
        base=seed_start+settings['suites'].index(suite)*1000
        if used&set(range(base,base+episodes)):raise ValueError('Evaluation seeds overlap train/validation')
        for mode in ['one_shot','feedback']:
            rows=[];latencies=[];sim=ArmSim(config['data']['image_size'])
            try:
                with (suite_root/f'{mode}.jsonl').open('w') as log:
                    for i in range(episodes):
                        seed=base+i;text=instruction_for(i,suite=='language');task=parse(text)
                        sim.reset(seed,suite if suite in ['lighting','camera','appearance','dynamics','positions'] else 'nominal')
                        if suite=='missing':sim.perturb('missing',task)
                        events=[]
                        def inject(stage):
                            if events:return
                            if (suite=='object_moved' and stage=='after_approach') or (suite=='slip' and stage=='after_lift'):
                                sim.perturb(suite,task);events.append({'stage':stage,'event':suite})
                        frames=[sim.image(384,True)]*10 if i<videos else []
                        commands_before=sim.command_count;agent=Agent(model,text,config['controller'])
                        result=agent.run(sim,mode,inject,frames if i<videos else None)
                        outcome=sim.score(task);moved_commands=sim.command_count-commands_before
                        if i<videos:frames.extend([sim.image(384,True)]*12)
                        row={'suite':suite,'mode':mode,'seed':seed,'instruction':text,'task':task.kind,'object':task.object,'pad':task.pad,
                            **outcome,**result,'events':events,'motion_segments':moved_commands,
                            'safe_stop':bool(result['status'].startswith('stopped') and moved_commands==0)}
                        log.write(json.dumps(row)+'\n');log.flush();rows.append(row);latencies.extend(result['inference_ms'])
                        if i<videos:annotate_video(suite_root/f'{mode}_{i:03d}.mp4',frames,text,f'{suite} | {mode} | {"success" if outcome["success"] else result["status"]}')
                summary={'suite':suite,'mode':mode,'episodes':episodes,'successes':sum(r['success'] for r in rows),
                    'safe_stops':sum(r['safe_stop'] for r in rows),'retries':sum(r['retries'] for r in rows),
                    'recovery_triggers':sum(r['recovery_triggered'] for r in rows),'checkpoint_sha256':digest,
                    'by_task':{kind:{'episodes':sum(r['task']==kind for r in rows),'successes':sum(r['success'] and r['task']==kind for r in rows)} for kind in ['place','lift']},
                    'by_object':{name:{'episodes':sum(r['object']==name for r in rows),'successes':sum(r['success'] and r['object']==name for r in rows)} for name in OBJECTS},
                    'latency_p50_ms':float(np.median(latencies)),'latency_p95_ms':float(np.percentile(latencies,95))}
                summary['success_rate']=summary['successes']/episodes;summary['wilson_95']=interval(summary['successes'],episodes)
                summary['safe_stop_rate']=summary['safe_stops']/episodes
                (suite_root/f'{mode}_summary.json').write_text(json.dumps(summary,indent=2));summaries.append(summary)
                print(json.dumps(summary),flush=True)
            finally:sim.close()
    result={'checkpoint':str(checkpoint),'checkpoint_sha256':digest,'parameters':sum(p.numel() for p in model.parameters()),
        'checkpoint_bytes':checkpoint.stat().st_size,'benchmark_seconds':time.time()-begin,'seed_start':seed_start,'suites':summaries}
    (output/'summary.json').write_text(json.dumps(result,indent=2));return result

def demo(config,run_root,instruction,seed=80000,event_name=None,output=None):
    root=Path(run_root);model,_=load_policy(root/'model/best.pt');torch.set_num_threads(config['training']['threads'])
    task=parse(instruction);sim=ArmSim(config['data']['image_size']);frames=[];events=[]
    try:
        sim.reset(seed)
        frames.extend([sim.image(384,True)]*12)
        def inject(stage):
            if not events and ((event_name=='object_moved' and stage=='after_approach') or (event_name=='slip' and stage=='after_lift')):
                sim.perturb(event_name,task);events.append(stage)
        result=Agent(model,instruction,config['controller']).run(sim,'feedback',inject,frames)
        outcome=sim.score(task);path=Path(output) if output else root/'demo.mp4'
        frames.extend([sim.image(384,True)]*20)
        annotate_video(path,frames,instruction,f'TinyVLA v2 | feedback | {event_name or "nominal"}')
        report={'instruction':instruction,'seed':seed,**outcome,**result,'event':event_name,'video':str(path)}
        path.with_suffix('.json').write_text(json.dumps(report,indent=2));return report
    finally:sim.close()

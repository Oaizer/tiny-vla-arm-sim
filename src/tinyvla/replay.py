"""Save videos that show how selected failed trials played out."""
import json
from pathlib import Path
import torch
from .sim import ArmSim
from .controller import Agent
from .language import parse
from .evaluate import load_policy,annotate_video

def replay_failures(config,root):
    root=Path(root);model,_=load_policy(root/'model/best.pt');torch.set_num_threads(config['training']['threads'])
    reports=[]
    for suite in config['evaluation']['suites']:
        if suite=='missing':continue
        for mode in ['one_shot','feedback']:
            source=root/'evaluation'/suite/f'{mode}.jsonl'
            if not source.exists():continue
            rows=[json.loads(line) for line in source.read_text().splitlines()]
            original=next((r for r in rows if not r['success']),None)
            if original is None:continue
            task=parse(original['instruction']);sim=ArmSim(config['data']['image_size']);events=[]
            try:
                sim.reset(original['seed'],suite if suite in ['lighting','camera','appearance','dynamics','positions'] else 'nominal')
                frames=[sim.image(384,True)]*12
                def inject(stage):
                    if not events and ((suite=='object_moved' and stage=='after_approach') or (suite=='slip' and stage=='after_lift')):
                        sim.perturb(suite,task);events.append(stage)
                result=Agent(model,task.instruction,config['controller']).run(sim,mode,inject,frames)
                outcome=sim.score(task);frames.extend([sim.image(384,True)]*20)
                path=root/'evaluation'/suite/f'{mode}_failure_{original["seed"]}.mp4'
                annotate_video(path,frames,task.instruction,f'{suite} | {mode} | failed trial replay')
                report={'suite':suite,'mode':mode,'seed':original['seed'],'original_success':original['success'],
                        **outcome,**result,'matches_original_success':outcome['success']==original['success'],'video':str(path)}
                path.with_suffix('.json').write_text(json.dumps(report,indent=2));reports.append(report)
                print(json.dumps({k:report[k] for k in ['suite','mode','seed','success','matches_original_success']}),flush=True)
            finally:sim.close()
    (root/'failure_replays.json').write_text(json.dumps(reports,indent=2));return reports

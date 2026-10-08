"""Save a plain-language results report and file hashes for each run."""
import hashlib,json,platform,subprocess,sys
from pathlib import Path
import torch
from . import __version__

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as file:
        for chunk in iter(lambda:file.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def write_report(config,root,result):
    root=Path(root);ckpt=torch.load(root/'model/best.pt',weights_only=False)
    project=Path(__file__).resolve().parents[2]
    try:commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=project,text=True,stderr=subprocess.DEVNULL).strip()
    except subprocess.CalledProcessError:commit=None
    sources={str(p.relative_to(project)):sha(p) for p in sorted((project/'src').rglob('*')) if p.is_file() and p.suffix in ['.py','.png']}
    files={str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*')) if p.is_file() and p.suffix in ['.pt','.npz','.jsonl']}
    manifest={'project_version':__version__,'git_commit':commit,'python':sys.version,'platform':platform.platform(),
        'torch':torch.__version__,'config':config,'sources_sha256':sources,'artifacts_sha256':files,
        'checkpoint_sha256':result['checkpoint_sha256'],'architecture':ckpt['architecture'],'seed_ranges':ckpt['seed_ranges']}
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2));(root/'config.json').write_text(json.dumps(config,indent=2))
    lines=['# TinyVLA v2 results','',
        f"The CPU model has **{result['parameters']:,} learned parameters**. Its checkpoint is **{result['checkpoint_bytes']:,} bytes**.",
        f"Training took {ckpt['training_seconds']:.1f} seconds. The horizontal action error on validation data was {ckpt['metrics']['xy_rmse_mm']:.2f} mm RMSE.",'',
        'The feedback controller checks a new image before it grasps and before it places an object. It uses the finger sensors to detect an empty grasp, retries a limited number of times, and then stops. These recovery rules are hand-written. The model learns to read images and commands and to predict arm waypoints.','',
        '| Test set | One-shot successes | Feedback successes | Feedback retries |','|---|---:|---:|---:|']
    names={'nominal':'Standard scenes','lighting':'Changed lighting','camera':'Changed camera view',
        'appearance':'Changed object size and angle','dynamics':'Changed object weight and grip',
        'positions':'Changed object positions','language':'New command wordings',
        'object_moved':'Object moved during the task','slip':'Object dropped after the lift'}
    for suite in config['evaluation']['suites']:
        rows={r['mode']:r for r in result['suites'] if r['suite']==suite}
        if len(rows)!=2:continue
        one,fb=rows['one_shot'],rows['feedback']
        if suite=='missing':
            lines.append(f"| Missing object: stopped before moving | {one['safe_stops']}/{one['episodes']} | {fb['safe_stops']}/{fb['episodes']} | {fb['retries']} |")
        else:lines.append(f"| {names.get(suite,suite)} | {one['successes']}/{one['episodes']} | {fb['successes']}/{fb['episodes']} | {fb['retries']} |")
    slip=next((r for r in result['suites'] if r['suite']=='slip' and r['mode']=='feedback'),None)
    if slip:
        breakdown=', '.join(f"{name}: {row['successes']}/{row['episodes']}" for name,row in slip['by_object'].items())
        lines += ['',f'Success after a forced slip, by object: {breakdown}. Dropped cylinders remain difficult to recover.']
    p95=max(r['latency_p95_ms'] for r in result['suites'])
    lines += ['',f'The slowest p95 model inference time across the test sets was {p95:.2f} ms on this CPU. This does not include rendering, arm calculations, or movement.',
        '', 'Each pair of results uses the same scene seeds and commands. After each trial, the scorer checks the simulator state. The model and controller cannot read this scoring data. In the moved-object test, the scene moves the object once after the arm approaches. In the slip test, the scene opens and closes the gripper once after the lift.','',
        'The JSON summaries show results by task and object, confidence intervals, model timing, and checkpoint hashes. The JSONL files keep each trial, command, retry, stop reason, and controller event. Videos show the first trials listed in the configuration.','',
        '## Scope','',
        'The task has four object types (red cube, red cylinder, blue cube, and green cylinder), three pad colors, and two task types. The command grammar and camera calibration are provided. The model does not receive object positions. Test sets change scene features that were also varied during training. This small model does not support open-ended language. We tested it only in simulation.','',
        'The v1 result of 94 successes in 100 trials came from easier scenes with different objects, pads, and camera settings. It cannot be compared directly with these v2 results.','',
        '## Reproducibility','',
        'See `manifest.json` for source and artifact hashes, the software environment, Git revision, settings, and seed ranges. Use the lock file and `tinyvla run` to create a new experiment. The Git repository includes this run and its data.']
    (root/'RESULTS.md').write_text('\n'.join(lines)+'\n');return manifest

"""Write a reviewable report and hash manifest alongside each experiment."""
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
    lines=['# TinyVLA v2 experiment results','',
        f"Trained CPU model: **{result['parameters']:,} parameters**, **{result['checkpoint_bytes']:,} checkpoint bytes**.",
        f"Training time: {ckpt['training_seconds']:.1f} seconds; validation horizontal action RMSE: {ckpt['metrics']['xy_rmse_mm']:.2f} mm.",'',
        'The feedback supervisor replans from fresh RGB before grasping and before placement, detects empty grasps using finger encoders, and retries within a fixed attempt budget. The recovery logic is scripted; perception, instruction selection, and waypoint heads are learned.','',
        '| Suite | One-shot success | Feedback success | Feedback retries |','|---|---:|---:|---:|']
    for suite in config['evaluation']['suites']:
        rows={r['mode']:r for r in result['suites'] if r['suite']==suite}
        if len(rows)!=2:continue
        one,fb=rows['one_shot'],rows['feedback']
        if suite=='missing':
            lines.append(f"| missing source (safe stop) | {one['safe_stops']}/{one['episodes']} | {fb['safe_stops']}/{fb['episodes']} | {fb['retries']} |")
        else:lines.append(f"| {suite} | {one['successes']}/{one['episodes']} | {fb['successes']}/{fb['episodes']} | {fb['retries']} |")
    slip=next((r for r in result['suites'] if r['suite']=='slip' and r['mode']=='feedback'),None)
    if slip:
        breakdown=', '.join(f"{name}: {row['successes']}/{row['episodes']}" for name,row in slip['by_object'].items())
        lines += ['',f'Slip recovery by object: {breakdown}. Fallen cylinders remain a substantial failure case.']
    p95=max(r['latency_p95_ms'] for r in result['suites'])
    lines += ['',f'Maximum per-suite p95 neural inference latency: {p95:.2f} ms on this CPU. This excludes rendering, inverse kinematics, and robot motion.',
        '', 'Each pair uses matching scene seeds and instructions. Success is measured by simulator state only after execution; that state is unavailable to the policy/supervisor. Missing-source rows measure stopping before any task motion. The moved-object event relocates the object once after approach; the slip event opens and recloses the gripper once after lifting.','',
        'Detailed JSON summaries include task/object breakdowns, Wilson confidence intervals, inference latency, and checkpoint hashes. JSONL files retain all trials, commands, supervisor events, retries, and stop reasons. Videos show the first configured trials per suite.','',
        '## Scope','',
        'Four object identities (red cube, red cylinder, blue cube, green cylinder), three pad destinations, two task families, and a declared command grammar. Camera calibration is supplied as an input; object coordinates are not. Images and dynamics vary within training ranges; individual benchmark suites apply held-out shifts. This is a compact structured visuomotor system, not open-vocabulary language or a foundation VLA. No physical arm was used.','',
        'The original 94/100 result describes a different, easier v1 scene distribution and is not directly comparable to these expanded v2 suites.','',
        '## Reproducibility','',
        'See `manifest.json` for the exact source/artifact hashes, environment, Git revision, configuration, and seed ranges. Use the repository lockfile and `tinyvla run` to reproduce a fresh experiment. Generated artifacts are excluded from Git but retained beside this report.']
    (root/'RESULTS.md').write_text('\n'.join(lines)+'\n');return manifest

"""One entrypoint for generation, training, paired evaluation, and demonstration."""
import argparse,json,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
from .data import generate
from .train import train
from .evaluate import benchmark,demo
from .report import write_report

def main():
    parser=argparse.ArgumentParser(prog='tinyvla');parser.add_argument('--config',default='configs/project.json')
    sub=parser.add_subparsers(dest='command',required=True)
    run=sub.add_parser('run',help='Generate, train, evaluate, and archive a fresh experiment')
    run.add_argument('--run-dir');run.add_argument('--skip-contract-checks',action='store_true')
    ev=sub.add_parser('evaluate');ev.add_argument('--run-dir',default='artifacts');ev.add_argument('--suites',nargs='+')
    ev.add_argument('--episodes',type=int);ev.add_argument('--videos',type=int);ev.add_argument('--seed-start',type=int);ev.add_argument('--output-name',default='evaluation')
    dm=sub.add_parser('demo');dm.add_argument('--run-dir',default='artifacts');dm.add_argument('--instruction',required=True)
    dm.add_argument('--seed',type=int,default=80000);dm.add_argument('--event',choices=['object_moved','slip']);dm.add_argument('--output')
    rep=sub.add_parser('report');rep.add_argument('--run-dir',default='artifacts')
    args=parser.parse_args();config=json.loads(Path(args.config).read_text())
    if args.command=='run':
        root=Path(args.run_dir or 'artifacts/runs/'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
        if root.exists():parser.error('Run directory already exists. Choose a new directory to preserve past results.')
        root.mkdir(parents=True);(root/'config.json').write_text(json.dumps(config,indent=2))
        project=Path(__file__).resolve().parents[2]
        if not args.skip_contract_checks:subprocess.run([sys.executable,'-m','unittest','discover','-s',str(project/'tests'),'-v'],check=True)
        data=config['data']
        gen={'train':generate(root/'data/train.npz',data['train_scenes'],data['train_seed_start'],data['image_size'],data['examples_per_scene']),
             'validation':generate(root/'data/validation.npz',data['validation_scenes'],data['validation_seed_start'],data['image_size'],data['examples_per_scene'])}
        (root/'generation.json').write_text(json.dumps(gen,indent=2));train(config,root)
        result=benchmark(config,root);demo(config,root,'put the red cylinder on the cyan pad',event_name='object_moved')
        write_report(config,root,result);print(f'Complete experiment: {root}/RESULTS.md',flush=True)
    elif args.command=='evaluate':
        if args.episodes is not None and args.episodes<1:parser.error('--episodes must be positive')
        if args.suites and not set(args.suites)<=set(config['evaluation']['suites']):parser.error('Unknown evaluation suite')
        benchmark(config,args.run_dir,args.suites,args.episodes,args.videos,args.seed_start,args.output_name)
    elif args.command=='demo':print(json.dumps(demo(config,args.run_dir,args.instruction,args.seed,args.event,args.output),indent=2))
    elif args.command=='report':
        root=Path(args.run_dir);write_report(config,root,json.loads((root/'evaluation/summary.json').read_text()))

if __name__=='__main__':main()

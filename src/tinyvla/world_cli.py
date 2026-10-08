"""Collect simulation data, train the world model, and compare controllers."""
import argparse
import json
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['collect', 'train', 'predict', 'evaluate', 'report', 'run'])
    parser.add_argument('--config', default='configs/world_model.json')
    parser.add_argument('--root', default='artifacts/world-v1')
    parser.add_argument('--policy', default='artifacts/model/best.pt')
    parser.add_argument('--split', choices=['train', 'validation', 'test'])
    parser.add_argument('--limit', type=int)
    parser.add_argument('--offset', type=int, default=0)
    parser.add_argument('--development', action='store_true')
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(config['training']['threads'])
    (root / 'config.json').write_text(json.dumps(config, indent=2))
    if args.command in ('collect', 'run'):
        from .world_data import collect
        torch.set_num_threads(1)
        for split in ([args.split] if args.split else ['train', 'validation', 'test']):
            collect(config, root, args.policy, split, args.limit, args.offset)
    if args.command in ('train', 'run'):
        from .world_train import train
        torch.set_num_threads(config['training']['threads'])
        train(config, root)
    if args.command in ('predict', 'run'):
        from .world_train import evaluate_prediction
        evaluate_prediction(config, root)
    if args.command in ('evaluate', 'run'):
        from .world_evaluate import benchmark
        torch.set_num_threads(config['evaluation'].get('inference_threads', 1))
        benchmark(config, root, args.policy, args.development, args.limit)
    if args.command in ('report', 'run'):
        from .world_report import report
        report(config, root)


if __name__ == '__main__':
    main()

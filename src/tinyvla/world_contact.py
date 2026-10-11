"""Collect direct contact data and evaluate a contact predictor offline."""
import argparse
import json
from pathlib import Path

from .world_contact_data import collect
from .world_contact_train import train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['collect', 'train'])
    parser.add_argument('--split', choices=['train', 'validation', 'test'])
    parser.add_argument('--count', type=int)
    parser.add_argument('--config', default='configs/world_contact.json')
    parser.add_argument('--root', default='artifacts/world-v3/contact')
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    root = Path(args.root)
    if args.command == 'collect':
        for split in ([args.split] if args.split else ['train', 'validation', 'test']):
            collect(config, root, config['source']['policy'], split, args.count)
    else:
        train(config, root)


if __name__ == '__main__':
    main()

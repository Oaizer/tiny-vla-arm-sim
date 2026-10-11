"""Verify version 5 reranker data, frozen inputs, and fresh ranking audits."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from tinyvla.world_contact_reranker import Ranker, scene_metrics

ROOT = Path(__file__).resolve().parents[1]
V5 = ROOT / 'artifacts/world-v5/reranker'
REP = ROOT / 'artifacts/world-v5/replication'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_sources(source_hashes):
    for relative, expected in source_hashes.items():
        assert sha(ROOT / relative) == expected, f'source changed: {relative}'


def inspect_episodes(folder, start, count):
    paths = sorted(folder.glob('*.npz'), key=lambda path: int(path.stem))
    assert [int(p.stem) for p in paths] == list(range(start, start + count)), folder
    rows = {}
    for path in paths:
        meta = json.loads(path.with_suffix('.json').read_text())
        assert meta['seed'] == int(path.stem)
        assert meta['object'] == 'red cylinder'
        with np.load(path) as record:
            assert len(record['features']) == len(record['actions']) + 1
            assert record['contact_force_n'].shape == (len(record['features']), 4)
        rows[path.name] = sha(path)
    return rows


def main():
    torch.set_num_threads(1)
    protocol = json.loads((V5 / 'protocol.json').read_text())
    assert protocol['candidate_count'] == 25
    assert protocol['torch_threads'] == 1
    check_sources(protocol['source_hashes'])
    assert protocol['contact_head_sha256'] == sha(ROOT / 'artifacts/world-v4/contact/contact-head.pt')

    inventory = {}
    for split, start, count in [('train', 188000, 40), ('validation', 188100, 20), ('test', 188200, 40)]:
        inventory[split] = inspect_episodes(V5 / 'episodes' / split, start, count)
        assert inventory[split] == protocol['episode_sha256'][split]
        with np.load(V5 / f'branches_{split}.npz') as data:
            x, y, seeds = data['features'], data['labels'], data['scene_seed']
        assert x.shape == (count * 25, 13) and y.shape == seeds.shape == (count * 25,)
        assert sorted(np.unique(seeds).tolist()) == list(range(start, start + count))
        assert np.all((y == 0) | (y == 1))
        assert sha(V5 / f'branches_{split}.npz') == protocol['branch_sha256'][split]

    bundle = torch.load(V5 / 'ranker.pt', map_location='cpu', weights_only=False)
    assert bundle['protocol_sha256'] == sha(V5 / 'protocol.json')
    test = json.loads((V5 / 'test_results.json').read_text())
    assert test['protocol_sha256'] == sha(V5 / 'protocol.json')
    assert test['ranker_sha256'] == sha(V5 / 'ranker.pt')
    assert test['test_scenes'] == 40 and test['test_branches'] == 1000
    model = Ranker(len(bundle['feature_names']), bundle['hidden_size'])
    model.load_state_dict(bundle['state_dict']); model.eval()
    with np.load(V5 / 'branches_test.npz') as data:
        x, y, seeds = data['features'], data['labels'], data['scene_seed']
    with torch.no_grad(): scores = model(torch.from_numpy((x - bundle['mean']) / bundle['std'])).numpy()
    calculated = scene_metrics(scores, y, seeds)
    assert abs(calculated['top1_contact_rate'] - test['test_metrics']['reranker']['top1_contact_rate']) < 1e-12
    assert abs(calculated['macro_average_precision'] - test['test_metrics']['reranker']['macro_average_precision']) < 1e-12

    replication = json.loads((REP / 'protocol.json').read_text())
    check_sources(replication['source_hashes'])
    assert replication['frozen_v5_protocol_sha256'] == sha(V5 / 'protocol.json')
    replication_inventory = inspect_episodes(REP / 'episodes' / 'test', 188240, 40)
    assert replication_inventory == replication['episode_sha256']
    assert sha(REP / 'branches.npz') == replication['branch_sha256']
    result = json.loads((REP / 'results.json').read_text())
    assert result['protocol_sha256'] == sha(REP / 'protocol.json')
    assert result['reranker_sha256'] == sha(V5 / 'ranker.pt')
    assert result['scenes'] == 40 and result['branches'] == 1000
    with np.load(REP / 'branches.npz') as data:
        x, y, seeds = data['features'], data['labels'], data['scene_seed']
    with torch.no_grad(): scores = model(torch.from_numpy((x - bundle['mean']) / bundle['std'])).numpy()
    calculated = scene_metrics(scores, y, seeds)
    assert abs(calculated['top1_contact_rate'] - result['test_metrics']['reranker']['top1_contact_rate']) < 1e-12
    assert abs(calculated['macro_average_precision'] - result['test_metrics']['reranker']['macro_average_precision']) < 1e-12

    manifest = {'experiment': 'world-v5-contact-reranker',
                'source_commit': __import__('subprocess').check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                'train_validation_test_scenes': {'train': 40, 'validation': 20, 'test': 40},
                'replication_scenes': 40, 'verified_branch_counts': {'train': 1000, 'validation': 500, 'test': 1000, 'replication': 1000},
                'files': {}}
    for path in sorted(ROOT.joinpath('artifacts/world-v5').rglob('*')):
        if path.is_file() and path.name != 'manifest.json':
            manifest['files'][path.relative_to(ROOT).as_posix()] = sha(path)
    out = ROOT / 'artifacts/world-v5/manifest.json'
    out.write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'verified': True, 'files': len(manifest['files']),
                      'v5_test': test['test_metrics'], 'replication': result['test_metrics'],
                      'manifest': 'artifacts/world-v5/manifest.json'}, indent=2))


if __name__ == '__main__':
    main()

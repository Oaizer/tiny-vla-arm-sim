"""Check version 4 episode records, evaluation outputs, and source hashes."""
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/world-v4/contact'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def verify_split(name, seed_start, count):
    folder = OUT / name
    files = sorted(folder.glob('*.npz'), key=lambda path: int(path.stem))
    assert [int(path.stem) for path in files] == list(range(seed_start, seed_start + count)), name
    positives = 0
    for path in files:
        meta = json.loads(path.with_suffix('.json').read_text())
        assert meta['object'] == 'red cylinder'
        assert meta['seed'] == int(path.stem)
        with np.load(path) as data:
            features, actions, times = data['features'], data['actions'], data['times']
            force = data['contact_force_n']
            assert len(features) == len(actions) + 1 == len(times)
            assert force.shape == (len(features), 4)
            assert np.allclose(np.diff(times), actions[:, 4], atol=1e-6)
            labels = force > .01
            assert int(labels.sum()) == meta['positive_steps']
            assert labels.sum(axis=0).tolist() == meta['contact_steps']
            positives += int((force[:, 3] > .01).sum())
    return {'episodes': len(files), 'positive_red_cylinder_observations': positives}


def main():
    splits = {'train': (185000, 103), 'validation': (185600, 24), 'test': (186200, 40), 'ranking': (187000, 20)}
    verified = {name: verify_split(name, start, count) for name, (start, count) in splits.items()}
    report = json.loads((OUT / 'report.json').read_text())
    rank = json.loads((OUT / 'ranking_results.json').read_text())
    protocol = json.loads((OUT / 'ranking_protocol.json').read_text())
    assert len(rank['rows']) == 20 and rank['branches'] == 500
    assert rank['scenes'] == len(rank['rows'])
    assert len({row['seed'] for row in rank['rows']}) == 20
    assert all(len(row['actual_contact']) == 25 for row in rank['rows'])
    assert abs(rank['head_top1_rate'] - np.mean([row['head_top1_contact'] for row in rank['rows']])) < 1e-12
    assert abs(rank['unchanged_rate'] - np.mean([row['unchanged_contact'] for row in rank['rows']])) < 1e-12
    for relative, expected in report['source_hashes'].items():
        assert sha(ROOT / relative) == expected, f'source changed: {relative}'
    assert sha(OUT / 'contact-head.pt') == report['checkpoint_sha256']
    assert protocol['contact_report_sha256'] == sha(OUT / 'report.json')
    for relative, expected in protocol['source_hashes'].items():
        assert sha(ROOT / relative) == expected, f'ranking input changed: {relative}'
    assert rank['protocol'] == protocol
    for split, hashes in report['dataset_sha256'].items():
        folder = OUT / split.removeprefix('target_') if split.startswith('target_') else Path('artifacts/world-v3/contact') / split.removeprefix('base_')
        for filename, expected in hashes.items():
            assert sha(ROOT / folder / filename) == expected, f'data changed: {split}/{filename}'
    manifest = {'experiment': 'world-v4-targeted-contact', 'verified': verified,
                'source_commit': __import__('subprocess').check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                'files': {}}
    for path in sorted(OUT.rglob('*')):
        if path.is_file() and path.name != 'manifest.json':
            manifest['files'][path.relative_to(ROOT).as_posix()] = sha(path)
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'verified': verified, 'ranking_scenes': 20, 'candidate_branches': 500,
                      'manifest': 'artifacts/world-v4/contact/manifest.json'}, indent=2))


if __name__ == '__main__':
    main()

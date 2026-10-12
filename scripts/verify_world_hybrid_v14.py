"""Check the frozen V14 source protocols and same-seed motion comparison."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'artifacts/world-v14'


def read_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha256(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_protocol(path: Path):
    protocol = read_json(path)
    for name, expected in protocol['source_sha256'].items():
        actual = sha256(ROOT / name)
        if actual != expected:
            raise SystemExit(f'Source hash mismatch: {name}')
    if protocol.get('model_sha256') and protocol['model_sha256'] != sha256(ROOT / 'artifacts/world-v13/task-ranker/ranker.pt'):
        raise SystemExit(f'Model hash mismatch: {path}')
    return protocol


original_protocol = check_protocol(RUN / 'original-motion/protocol.json')
direct_protocol = check_protocol(RUN / 'direct-approach/protocol.json')
if original_protocol['seed_start'] != direct_protocol['seed_start']:
    raise SystemExit('The original and direct tests use different starting seeds.')

original_root = RUN / 'original-motion/evaluation'
direct_root = RUN / 'direct-approach/evaluation'
task_suites = ['nominal', 'tipped', 'slip', 'object_moved', 'dynamics']
hybrid_rows = []
for suite in task_suites + ['missing']:
    old = {row['seed']: row for row in map(json.loads, (original_root / suite / 'hybrid.jsonl').read_text().splitlines())}
    new = {row['seed']: row for row in map(json.loads, (direct_root / suite / 'hybrid.jsonl').read_text().splitlines())}
    if old.keys() != new.keys():
        raise SystemExit(f'Unmatched hybrid seeds in {suite}.')
    for seed, old_row in old.items():
        new_row = new[seed]
        if suite != 'missing':
            hybrid_rows.append((old_row, new_row))
            if old_row['success'] != new_row['success']:
                raise SystemExit(f'Hybrid task outcome changed for {suite}, seed {seed}.')
        elif not old_row['safe_stop'] or not new_row['safe_stop']:
            raise SystemExit(f'Missing-object safe stop failed for seed {seed}.')
        if suite == 'nominal' and seed == original_protocol['seed_start']:
            if 'pregrasp_replan' not in [entry['phase'] for entry in old_row['trace']]:
                raise SystemExit('Original motion trace did not contain its pregrasp replan.')
            if 'pregrasp_replan' in [entry['phase'] for entry in new_row['trace']]:
                raise SystemExit('Direct motion trace still contains the removed pregrasp replan.')

old_calls = sum(row['inference_calls'] for suite in task_suites
                for row in map(json.loads, (original_root / suite / 'hybrid.jsonl').read_text().splitlines()))
new_calls = sum(row['inference_calls'] for suite in task_suites
                for row in map(json.loads, (direct_root / suite / 'hybrid.jsonl').read_text().splitlines()))
if len(hybrid_rows) != 40 or old_calls != 122 or new_calls != 89:
    raise SystemExit(f'Unexpected V14 trial or inference counts: {len(hybrid_rows)}, {old_calls}, {new_calls}.')

print(json.dumps({
    'verified': True,
    'task_trials': len(hybrid_rows),
    'hybrid_paired_wins': 0,
    'hybrid_paired_losses': 0,
    'hybrid_paired_ties': len(hybrid_rows),
    'missing_safe_stops': 8,
    'original_policy_calls': old_calls,
    'direct_policy_calls': new_calls,
    'source_protocols': 2,
}, indent=2))

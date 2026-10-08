"""Verify the frozen version 2 evidence before publishing it."""
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


root = Path('artifacts/world-v2')
protocol = json.loads((root / 'FINAL_PROTOCOL.json').read_text())
for name, expected in protocol['hashes'].items():
    if sha(Path(name)) != expected:
        raise ValueError(f'Frozen input changed: {name}')
summary = json.loads((root / 'evaluation/summary.json').read_text())
if summary['config'] != protocol['config']:
    raise ValueError('The run configuration differs from the protocol.')
settings = protocol['config']['evaluation']
total = 0
for index, suite in enumerate(settings['suites']):
    expected_seeds = list(range(settings['final_seed'] + index * 1000,
                                settings['final_seed'] + index * 1000 + settings['episodes']))
    instructions = None
    for mode in ['feedback', 'jepa', 'coherent']:
        path = root / 'evaluation' / suite / f'{mode}.jsonl'
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if [r['seed'] for r in rows] != expected_seeds:
            raise ValueError(f'Wrong or missing trial seeds: {path}')
        trial_instructions = [r['instruction'] for r in rows]
        if instructions is not None and trial_instructions != instructions:
            raise ValueError(f'Tasks do not match: {path}')
        instructions = trial_instructions
        if any(r['error'] is not None for r in rows):
            raise ValueError(f'Controller error in {path}')
        record = next(s for s in summary['suites'] if s['mode'] == mode and s['suite'] == suite)
        if sum(r['success'] for r in rows) != record['successes']:
            raise ValueError(f'Success count mismatch: {path}')
        if sum(r['safe_stop'] for r in rows) != record['safe_stops']:
            raise ValueError(f'Stop count mismatch: {path}')
        total += len(rows)
audit = json.loads((root / 'prediction_audit.json').read_text())
if len(audit) != 7 or any(r['repeat_max_error_m'] > 1e-6 for r in audit):
    raise ValueError('Incomplete or nonrepeatable candidate audit.')
manifest = {'verified_trials': total, 'verified_audit_cases': len(audit),
            'source_commit': protocol['source_commit'], 'frozen_inputs_match': True,
            'files': {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob('*'))
                      if p.is_file() and p.name != 'manifest.json'}}
(root / 'manifest.json').write_text(json.dumps(manifest, indent=2))
print(json.dumps({k: v for k, v in manifest.items() if k != 'files'}))

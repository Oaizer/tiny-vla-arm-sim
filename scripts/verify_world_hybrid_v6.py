"""Verify the frozen V6 hybrid-controller comparison and its outputs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'artifacts/world-v6/hybrid-feedback'


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    protocol_path = RUN / 'protocol.json'
    protocol = json.loads(protocol_path.read_text())
    for relative, expected in protocol['source_sha256'].items():
        assert sha(ROOT / relative) == expected, f'source changed: {relative}'

    config = json.loads((ROOT / 'configs/world_hybrid_v6.json').read_text())
    all_rows = []
    for suite in config['evaluation']['suites']:
        folder = RUN / 'evaluation' / suite
        feedback = [json.loads(line) for line in (folder / 'feedback.jsonl').read_text().splitlines()]
        hybrid = [json.loads(line) for line in (folder / 'hybrid.jsonl').read_text().splitlines()]
        expected = list(range(config['evaluation']['final_seed'] + config['evaluation']['suites'].index(suite) * 1000,
                              config['evaluation']['final_seed'] + config['evaluation']['suites'].index(suite) * 1000 + config['evaluation']['episodes']))
        assert [row['seed'] for row in feedback] == expected
        assert [row['seed'] for row in hybrid] == expected
        assert all(row['object'] == 'red cylinder' for row in feedback + hybrid)
        assert (folder / 'feedback.mp4').is_file() and (folder / 'hybrid.mp4').is_file()
        all_rows.extend((suite, left, right) for left, right in zip(feedback, hybrid))

    task_rows = [(suite, feedback, hybrid) for suite, feedback, hybrid in all_rows if suite != 'missing']
    wins = sum(hybrid['success'] and not feedback['success'] for _, feedback, hybrid in task_rows)
    losses = sum(feedback['success'] and not hybrid['success'] for _, feedback, hybrid in task_rows)
    ties = len(task_rows) - wins - losses
    assert len(task_rows) == 40 and (wins, losses, ties) == (0, 0, 40)
    assert sum(row['safe_stop'] for suite, row, _ in all_rows if suite == 'missing') == 7
    assert sum(row['safe_stop'] for suite, _, row in all_rows if suite == 'missing') == 7
    decisions = [decision for _, _, row in all_rows for decision in row['planning']]
    assert len(decisions) == 3 and all(row['accepted'] for row in decisions)

    paired = json.loads((RUN / 'paired_results.json').read_text())
    summary = json.loads((RUN / 'summary.json').read_text())
    assert paired['feedback_successes'] == paired['hybrid_successes'] == 26
    assert (paired['wins'], paired['losses'], paired['ties']) == (0, 0, 40)
    assert summary['protocol_sha256'] == sha(protocol_path)
    assert all((RUN / 'evaluation' / suite / f'{mode}.jsonl').is_file()
               for suite in config['evaluation']['suites'] for mode in ('feedback', 'hybrid'))
    print(json.dumps({'verified': True, 'task_trials': len(task_rows),
                      'feedback_successes': 26, 'hybrid_successes': 26,
                      'paired': {'wins': wins, 'losses': losses, 'ties': ties},
                      'ranker_proposals': len(decisions), 'protocol_sha256': sha(protocol_path)}, indent=2))


if __name__ == '__main__':
    main()

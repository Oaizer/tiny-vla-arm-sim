"""Check V13 source hashes, matched seeds, outcomes, and videos."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'artifacts/world-v13/task-ranker'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_protocol(path):
    protocol=json.loads(path.read_text())
    for relative,expected in protocol['source_sha256'].items():
        assert sha(ROOT/relative)==expected,f'source hash changed: {relative}'
    return protocol


def rows_for(folder,suite,expected_seeds):
    source=folder/'evaluation'/suite
    feedback=[json.loads(line) for line in (source/'feedback.jsonl').read_text().splitlines()]
    hybrid=[json.loads(line) for line in (source/'hybrid.jsonl').read_text().splitlines()]
    assert [row['seed'] for row in feedback]==expected_seeds
    assert [row['seed'] for row in hybrid]==expected_seeds
    assert all(row['object']=='red cylinder' for row in feedback+hybrid)
    assert (source/'feedback.mp4').is_file() and (source/'hybrid.mp4').is_file()
    return list(zip(feedback,hybrid))


def summarize(pairs):
    task=[(left,right) for suite,left,right in pairs if suite!='missing']
    wins=sum(right['success'] and not left['success'] for left,right in task)
    losses=sum(left['success'] and not right['success'] for left,right in task)
    ties=len(task)-wins-losses
    return {'task_trials':len(task),'feedback_successes':sum(left['success'] for left,_ in task),
      'hybrid_successes':sum(right['success'] for _,right in task),'wins':wins,'losses':losses,'ties':ties}


def verify_run(config,root,seed_start):
    pairs=[]
    for index,suite in enumerate(config['evaluation']['suites']):
        first=seed_start+index*1000;seeds=list(range(first,first+config['evaluation']['episodes']))
        pairs.extend((suite,left,right) for left,right in rows_for(root,suite,seeds))
    return pairs


def main():
    offline=verify_protocol(RUN/'protocol.json')
    primary=verify_protocol(RUN/'controller_protocol_v2.json')
    replicate=verify_protocol(RUN/'replication_protocol.json')
    config=json.loads((ROOT/'configs/world_hybrid_v13.json').read_text())
    assert sha(RUN/'ranker.pt')==primary['offline_model_sha256']==replicate['ranker_sha256']
    assert primary['offline_protocol_sha256']==sha(RUN/'protocol.json')
    first=verify_run(config,RUN/'controller-v2',340000)
    second=verify_run(config,RUN/'replication',350000)
    a=summarize(first);b=summarize(second)
    assert (a['feedback_successes'],a['hybrid_successes'],a['wins'],a['losses'],a['ties'])==(26,30,8,4,28)
    assert (b['feedback_successes'],b['hybrid_successes'],b['wins'],b['losses'],b['ties'])==(25,27,3,1,36)
    combined=json.loads((RUN/'combined_results.json').read_text())
    assert combined['feedback_successes']==51 and combined['hybrid_successes']==57
    assert combined['wins']==11 and combined['losses']==5 and combined['ties']==64
    print(json.dumps({'verified':True,'same_ranker':True,'primary':a,'replication':b,
      'combined':combined,'offline_protocol_sha256':sha(RUN/'protocol.json'),
      'controller_protocol_sha256':sha(RUN/'controller_protocol_v2.json'),
      'replication_protocol_sha256':sha(RUN/'replication_protocol.json')},indent=2))


if __name__=='__main__':main()

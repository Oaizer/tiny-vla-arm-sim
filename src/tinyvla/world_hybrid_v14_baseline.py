"""Run the original V13 motion sequence on V14's matched seed range."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import torch

from . import world_evaluate
from .world_hybrid_corrected import CorrectedHybridGraspPlanner

ROOT = Path('artifacts/world-v14/original-motion')
CONFIG = Path('configs/world_hybrid_v14.json')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    cfg = json.loads(CONFIG.read_text())
    cfg['model_stages'] = {'hybrid': 'jepa'}
    cfg['evaluation']['final_seed'] = cfg['ranking']['controller_seed']
    ROOT.mkdir(parents=True, exist_ok=True)
    protocol_path = ROOT / 'protocol.json'
    if protocol_path.exists() or (ROOT / 'evaluation').exists():
        raise FileExistsError('V14 original-motion baseline already exists.')
    files = [CONFIG, Path('configs/project.json'), Path('uv.lock'),
             Path('src/tinyvla/world_hybrid_v14_baseline.py'),
             Path('src/tinyvla/world_hybrid_v13_controller.py'),
             Path('src/tinyvla/world_hybrid_v13_experiment.py'),
             Path('src/tinyvla/world_hybrid_corrected.py'),
             Path('src/tinyvla/world_hybrid_ranker_v7.py'), Path('src/tinyvla/world_evaluate.py'),
             Path('src/tinyvla/world_planner.py'), Path('src/tinyvla/controller.py'),
             Path('src/tinyvla/world_model.py'), Path('src/tinyvla/world_contact_rank.py'),
             Path('src/tinyvla/world_contact_reranker.py'), Path('src/tinyvla/world_contact_train.py'),
             Path('src/tinyvla/world_data.py'), Path('src/tinyvla/evaluate.py'), Path('src/tinyvla/sim.py'),
             Path('src/tinyvla/language.py'), Path('artifacts/model/best.pt'),
             Path('artifacts/world-v1/model/jepa.pt'), Path('artifacts/world-v4/contact/contact-head.pt'),
             Path('artifacts/world-v13/task-ranker/ranker.pt'),
             Path('artifacts/world-v13/task-ranker/controller_protocol_v2.json')]
    protocol = {
        'version': 1,
        'created_unix': int(time.time()),
        'seed_start': cfg['evaluation']['final_seed'],
        'episodes_per_condition': cfg['evaluation']['episodes'],
        'conditions': cfg['evaluation']['suites'],
        'change': 'Matched baseline: keep the original V13 approach, retreat-to-park, and pregrasp-replan sequence.',
        'model_sha256': digest(Path(cfg['hybrid']['reranker'])),
        'source_sha256': {str(path): digest(path) for path in files},
    }
    protocol_path.write_text(json.dumps(protocol, indent=2) + '\n')
    world_evaluate.HybridGraspPlanner = CorrectedHybridGraspPlanner
    result = world_evaluate.benchmark(cfg, ROOT, 'artifacts/model/best.pt')
    print(json.dumps({'evaluation_seconds': result['seconds'], 'evaluation': str(ROOT / 'evaluation'),
                      'protocol_sha256': digest(protocol_path)}, indent=2), flush=True)


if __name__ == '__main__':
    run()


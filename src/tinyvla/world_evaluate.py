"""Compare the existing controller and the world planner on matching trials."""
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from .controller import Agent
from .evaluate import annotate_video, instruction_for, interval, load_policy
from .language import parse
from .sim import ArmSim
from .world_data import VisualFeatures, tip_object
from .world_model import load_world
from .world_planner import GraspPlanner, PlanningAgent, SensorArm


def benchmark(config, root, policy_path, development=False, limit=None):
    root = Path(root)
    directory = root / (config.get('development_name', 'development') if development else 'evaluation')
    directory.mkdir(exist_ok=True)
    policy, _ = load_policy(policy_path)
    base_settings = json.loads(Path('configs/project.json').read_text())['controller']
    settings = config['evaluation']
    count = limit or (16 if development else settings['episodes'])
    start = settings['development_seed'] if development else settings['final_seed']
    extractor = VisualFeatures(policy)
    modes = ['feedback', 'fixed', 'jepa'] if development else ['feedback', *config.get('final_models', ['fixed', 'jepa'])]
    summaries = []
    begin = time.time()
    suites = config.get('development_suites', settings['suites']) if development else settings['suites']
    for suite in suites:
        suite_index = settings['suites'].index(suite)
        for mode in modes:
            model, ckpt = (None, None) if mode == 'feedback' else load_world(root / 'model' / f'{mode}.pt')
            first_seed = start + suite_index * 1000
            if ckpt and set(range(first_seed, first_seed + count)) & set(ckpt['training_seeds'] + ckpt['validation_seeds']):
                raise ValueError('Controller tests share training or validation seeds.')
            rows = []
            sim = ArmSim()
            folder = directory / suite
            folder.mkdir(exist_ok=True)
            try:
                with (folder / f'{mode}.jsonl').open('w') as log:
                    for index in range(count):
                        seed = first_seed + index
                        task = parse(instruction_for(index))
                        sim.reset(seed, 'dynamics' if suite == 'dynamics' else 'nominal')
                        if suite == 'tipped':
                            tip_object(sim, task)
                        if suite == 'missing':
                            sim.perturb('missing', task)
                        planner = None if model is None else GraspPlanner(model, config['planner'])
                        view = SensorArm(sim, extractor, planner)
                        agent = Agent(policy, task.instruction, base_settings) if planner is None else PlanningAgent(policy, task.instruction, base_settings)
                        events = []
                        def inject(stage):
                            if not events and ((suite == 'slip' and stage == 'after_lift') or
                                               (suite == 'object_moved' and stage == 'after_approach')):
                                sim.perturb(suite, task)
                                view.refresh()
                                events.append(stage)
                        frames = [sim.image(384, True)] * 10 if index == 0 else None
                        before = sim.command_count
                        tick = time.perf_counter()
                        result = agent.run(view, 'feedback', inject, frames)
                        elapsed = time.perf_counter() - tick
                        outcome = sim.score(task)
                        row = {'suite': suite, 'mode': mode, 'seed': seed, 'instruction': task.instruction,
                               'object': task.object, 'task': task.kind, **result, **outcome,
                               'safe_stop': result['status'].startswith('stopped') and sim.command_count == before,
                               'control_seconds': elapsed, 'feature_ms': view.feature_ms,
                               'planning': planner.decisions if planner else []}
                        log.write(json.dumps(row) + '\n'); log.flush()
                        rows.append(row)
                        if frames is not None:
                            frames.extend([sim.image(384, True)] * 20)
                            annotate_video(folder / f'{mode}.mp4', frames, task.instruction,
                                           f'{suite} | {mode} | {"success" if outcome["success"] else "failed"}')
                decisions = [d for row in rows for d in row['planning']]
                successes = sum(row['success'] for row in rows)
                summary = {'suite': suite, 'mode': mode, 'episodes': count, 'successes': successes,
                           'safe_stops': sum(row['safe_stop'] for row in rows), 'wilson_95': interval(successes, count),
                           'retries': sum(row['retries'] for row in rows), 'planner_calls': len(decisions),
                           'changed_commands': sum(d['accepted'] for d in decisions),
                           'planner_p95_ms': float(np.percentile([d['milliseconds'] for d in decisions], 95)) if decisions else None,
                           'feature_p95_ms': float(np.percentile([v for row in rows for v in row['feature_ms']], 95)),
                           'mean_control_seconds': float(np.mean([r['control_seconds'] for r in rows])),
                           'by_object': {name: {'episodes': sum(r['object'] == name for r in rows),
                                                'successes': sum(r['object'] == name and r['success'] for r in rows)}
                                         for name in sorted(set(r['object'] for r in rows))}}
                summaries.append(summary)
                (folder / f'{mode}_summary.json').write_text(json.dumps(summary, indent=2))
                print(json.dumps(summary), flush=True)
            finally:
                sim.close()
    result = {'development': development, 'seconds': time.time() - begin, 'suites': summaries,
              'policy_sha256': hashlib.sha256(Path(policy_path).read_bytes()).hexdigest(),
              'model_hashes': {mode: hashlib.sha256((root / 'model' / f'{mode}.pt').read_bytes()).hexdigest()
                               for mode in modes if mode != 'feedback'}, 'config': config}
    (directory / 'summary.json').write_text(json.dumps(result, indent=2))
    return result

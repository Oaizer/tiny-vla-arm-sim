"""Check candidate predictions with offline simulator branches on old failures.

Simulator labels are used only here for diagnosis. They never choose a command
for the evaluated controller. This audit uses version 1 seeds, not the new test.
"""
import json
from pathlib import Path
import numpy as np
import pybullet as p
import torch
from .evaluate import load_policy
from .language import OBJECTS, parse
from .sim import ArmSim
from .world_data import VisualFeatures, tip_object, training_target, XYZ_OFFSET, XYZ_SCALE
from .world_model import load_world
from .world_planner import GraspPlanner, PlanningAgent, SensorArm
from .world_iteration import paired


class SnapshotReached(Exception):
    pass


class AuditPlanner(GraspPlanner):
    def choose(self, *args):
        result = super().choose(*args)
        raise SnapshotReached()


def main():
    torch.set_num_threads(1)
    root = Path('artifacts/world-v2')
    root.mkdir(exist_ok=True)
    config = json.loads(Path('configs/world_model.json').read_text())
    settings = json.loads(Path('configs/project.json').read_text())['controller']
    policy, _ = load_policy('artifacts/model/best.pt')
    extractor = VisualFeatures(policy)
    model, _ = load_world('artifacts/world-v1/model/jepa.pt')
    comparison = paired('artifacts/world-v1/evaluation', 'jepa')
    results = []
    for case in comparison['wins'] + comparison['losses']:
        path = Path('artifacts/world-v1/evaluation') / case['suite'] / 'jepa.jsonl'
        original = next(r for r in map(json.loads, path.read_text().splitlines()) if r['seed'] == case['seed'])
        task = parse(original['instruction'])
        sim = ArmSim()
        try:
            sim.reset(case['seed'])
            if case['suite'] == 'tipped':
                tip_object(sim, task)
            planner = AuditPlanner(model, config['planner'])
            view = SensorArm(sim, extractor, planner)
            events = []
            def inject(stage):
                if not events and case['suite'] == 'slip' and stage == 'after_lift':
                    sim.perturb('slip', task)
                    view.refresh()
                    events.append(stage)
            try:
                PlanningAgent(policy, task.instruction, settings).run(view, 'feedback', inject)
            except SnapshotReached:
                pass
            else:
                raise AssertionError('No planning state reached on an intervention trial.')
            state_id = p.saveState()
            ticks, commands, heights = sim.ticks, sim.command_count, sim.max_heights.copy()
            def restore():
                p.restoreState(state_id)
                sim.ticks, sim.command_count, sim.max_heights = ticks, commands, heights.copy()
            measured = []
            src = OBJECTS.index(task.object)
            for sequence in planner.last_actions:
                restore()
                for action in sequence:
                    sim.move(action[:3], action[3], round(float(action[4]) * 240))
                target = training_target(sim)[src * 8:src * 8 + 8]
                pos = target[:3] * XYZ_SCALE + XYZ_OFFSET
                distance = float(np.linalg.norm(pos - sim.state()[:3]) / .1)
                height = float(np.clip((pos[2] - .02) / .1, 0, 1))
                workspace = bool(np.all(pos >= [.25, -.4, 0]) and np.all(pos <= [.8, .4, .4]))
                cost = distance - 2 * float(target[7]) - height + 2 * (not workspace)
                measured.append({'position': pos.tolist(), 'held': float(target[7]), 'cost': cost})
            # A repeated baseline branch checks that restore reproduces the physics.
            restore()
            for action in planner.last_actions[0]:
                sim.move(action[:3], action[3], round(float(action[4]) * 240))
            repeat = training_target(sim)[src * 8:src * 8 + 3] * XYZ_SCALE + XYZ_OFFSET
            repeat_error = float(np.max(np.abs(repeat - measured[0]['position'])))
            if repeat_error > 1e-6:
                raise AssertionError(f'Branch replay mismatch: {repeat_error}')
            predictions = planner.last_predictions[:, :, -1, src * 8:src * 8 + 8].mean(0)
            predicted_xyz = predictions[:, :3] * XYZ_SCALE + XYZ_OFFSET
            observed_xyz = np.array([r['position'] for r in measured])
            selected = planner.decisions[-1]['candidate']
            row = {'seed': case['seed'], 'suite': case['suite'], 'object': task.object,
                   'original_paired_outcome': 'win' if case['candidate_success'] else 'loss',
                   'selected': selected, 'selected_actual_cost_gain': measured[0]['cost'] - measured[selected]['cost'],
                   'best_actual_candidate': int(np.argmin([r['cost'] for r in measured])),
                   'position_rmse_mm': float(np.sqrt(np.mean((predicted_xyz - observed_xyz)**2)) * 1000),
                   'held_mae': float(np.mean(np.abs(predictions[:, 7].clip(0, 1) - [r['held'] for r in measured]))),
                   'repeat_max_error_m': repeat_error, 'decision': planner.decisions[-1],
                   'predicted_positions': predicted_xyz.tolist(), 'predicted_held': predictions[:, 7].clip(0, 1).tolist(),
                   'actual': measured, 'actions': planner.last_actions.tolist()}
            results.append(row)
            print(json.dumps({k: row[k] for k in ['seed', 'position_rmse_mm', 'held_mae', 'selected_actual_cost_gain']}), flush=True)
            p.removeState(state_id)
        finally:
            sim.close()
    (root / 'prediction_audit.json').write_text(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()

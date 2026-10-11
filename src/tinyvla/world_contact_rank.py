"""Collect fresh scenes and audit grasp candidate ranking without a controller."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pybullet as p
import torch

from .evaluate import load_policy
from .language import OBJECTS, parse
from .sim import ArmSim
from .world_contact_data import collect, source_hashes
from .world_contact_train import ContactHead
from .world_data import STATE_OFFSET, STATE_SCALE, tip_object
from .world_model import load_world


def candidates(anchor):
    offsets = [(x, y, z) for z in [-.012, -.006, 0, .006, .012]
               for x, y in [(0, 0), (-.006, 0), (.006, 0), (0, -.006), (0, .006)]]
    offsets = [(0, 0, 0)] + [value for value in offsets if value != (0, 0, 0)]
    return np.clip(np.asarray(anchor)[None] + np.asarray(offsets),
                   [.28, -.33, .015], [.78, .37, .38])


def average_precision(scores, labels):
    scores, labels = np.asarray(scores), np.asarray(labels, bool)
    positive = int(labels.sum())
    if positive == 0:
        return float('nan')
    order = np.argsort(-scores)
    hits = labels[order]
    precision = np.cumsum(hits) / np.arange(1, len(hits) + 1)
    return float((precision * hits).sum() / positive)


def main():
    torch.set_num_threads(1)
    root = Path('artifacts/world-v3/contact')
    data_config = json.loads(Path('configs/world_contact.json').read_text())
    model_path = Path(data_config['source']['checkpoint'])
    head_path = root / 'contact-head.pt'
    output_protocol = root / 'candidate_rank_protocol.json'
    if output_protocol.exists():
        raise FileExistsError('This ranking protocol already ran. Keep its results and use a new project copy for another study.')
    if not head_path.is_file() or not root.joinpath('report.json').is_file():
        raise FileNotFoundError('Collect and train the contact head before ranking candidates.')
    # Use new scene seeds that did not train or evaluate contact probabilities.
    ranking = {'episodes': 40, 'seed_base': 183000, 'index_offset': 400,
               'seed_start': 183400, 'seed_end': 183439, 'candidates_per_scene': 25,
               'steps_predicted': 3, 'steps_executed_for_contact': 2,
               'contact_threshold_n': .01}
    code_paths = [Path(data_config['source']['checkpoint']), Path(data_config['source']['policy']),
                  Path('configs/world_contact.json'), Path('src/tinyvla/world_contact_data.py'),
                  Path('src/tinyvla/world_contact_train.py'), Path('src/tinyvla/world_contact_rank.py'),
                  Path('src/tinyvla/world_model.py'), Path('src/tinyvla/world_data.py'),
                  Path('src/tinyvla/sim.py'), Path('src/tinyvla/evaluate.py'),
                  Path('src/tinyvla/controller.py'), Path('uv.lock')]
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in code_paths}
    hashes[str(head_path)] = hashlib.sha256(head_path.read_bytes()).hexdigest()
    protocol = {'ranking': ranking, 'model_hashes': hashes,
                'contact_test_report_sha256': hashlib.sha256((root / 'report.json').read_bytes()).hexdigest(),
                'warning': 'Offline candidate contact audit only. No planner controls the arm.'}
    output_protocol.write_text(json.dumps(protocol, indent=2))

    collect_config = json.loads(json.dumps(data_config))
    collect_config['data'].update({'ranking_episodes': ranking['episodes'],
                                   'ranking_seed': ranking['seed_base'],
                                   'ranking_offset': ranking['index_offset']})
    collect(collect_config, root, data_config['source']['policy'], 'ranking', ranking['episodes'])

    base, _ = load_world(model_path)
    bundle = torch.load(head_path, map_location='cpu', weights_only=False)
    head = ContactHead(bundle['latent_size'], bundle['hidden_size'])
    head.load_state_dict(bundle['state_dict']); head.eval()
    scale = torch.tensor(bundle['calibration_scale'])
    bias = torch.tensor(bundle['calibration_bias'])
    scenarios, seen = [], set()
    for path in sorted((root / 'ranking').glob('*.npz')):
        with np.load(path) as stored:
            features = stored['features'].astype(np.float32)
            actions = stored['actions'].astype(np.float32)
            seed = int(stored['seed'])
        metadata = json.loads(path.with_suffix('.json').read_text())
        task = parse(metadata['instruction'])
        source = OBJECTS.index(task.object)
        choice = next((index for index in range(2, len(actions) - 3)
                       if actions[index, 2] <= .055 and actions[index, 3] > .02
                       and seed not in seen), None)
        if choice is None:
            continue
        seen.add(seed)
        history = torch.from_numpy(features[choice - 2:choice + 1])[None]
        xyz = candidates(actions[choice, :3].copy())
        sequence = np.zeros((len(xyz), 3, 5), np.float32)
        sequence[:, 0, :3] = xyz
        sequence[:, 0, 3:] = [.04, 150 / 240]
        sequence[:, 1, :3] = xyz
        sequence[:, 1, 3:] = [0, 190 / 240]
        sequence[:, 2, :3] = xyz + [0, 0, .03]
        sequence[:, 2, 3:] = [0, 24 / 240]
        with torch.no_grad():
            latent = base.encode(history).expand(len(xyz), -1)
            states, outcomes = base.rollout(latent, torch.from_numpy(sequence))
            logits = head(states.permute(1, 0, 2, 3)).permute(1, 0, 2, 3)
            probability = (logits.mean(0)[:, 1, source] * scale[source] + bias[source]).sigmoid().numpy()
            old_held = outcomes[:, :, 1, source * 8 + 7].mean(0).numpy().clip(0, 1)

        sim = ArmSim()
        labels = []
        try:
            sim.reset(seed)
            if metadata['tipped']:
                tip_object(sim, task)
            for action in actions[:choice]:
                sim.move(action[:3], action[3], round(float(action[4]) * 240))
            state_error = float(np.max(np.abs((sim.state() - STATE_OFFSET) / STATE_SCALE - features[choice, -4:])))
            if state_error > .005:
                raise ValueError(f'Saved scene replay error {state_error:.5f} exceeds the protocol limit.')
            state_id = p.saveState()
            ticks, command_count, max_heights = sim.ticks, sim.command_count, sim.max_heights.copy()
            for candidate in sequence:
                p.restoreState(state_id)
                sim.ticks, sim.command_count, sim.max_heights = ticks, command_count, max_heights.copy()
                for action_index, action in enumerate(candidate):
                    sim.move(action[:3], action[3], round(float(action[4]) * 240))
                    if action_index == ranking['steps_executed_for_contact'] - 1:
                        break
                force = sum(max(0., float(point[9])) for point in
                            p.getContactPoints(bodyA=sim.arm, bodyB=sim.objects[task.object])
                            if point[3] in (9, 10))
                labels.append(force > ranking['contact_threshold_n'])
            p.removeState(state_id)
        finally:
            sim.close()
        labels = np.asarray(labels, bool)
        selected = int(probability.argmax())
        old_selected = int(old_held.argmax())
        scenarios.append({'seed': seed, 'object': task.object, 'instruction': task.instruction,
                          'replay_state_max_error': state_error,
                          'positive_candidate_count': int(labels.sum()),
                          'contact_head_top1_contact': bool(labels[selected]),
                          'old_held_top1_contact': bool(labels[old_selected]),
                          'unchanged_top1_contact': bool(labels[0]),
                          'oracle_has_contact': bool(labels.any()),
                          'contact_head_top1_index': selected,
                          'contact_head_probability': float(probability[selected]),
                          'contact_head_average_precision': average_precision(probability, labels),
                          'old_held_average_precision': average_precision(old_held, labels),
                          'contact_predictions': probability.tolist(), 'old_held_scores': old_held.tolist(),
                          'actual_contact': labels.tolist(), 'candidate_xyz': xyz.tolist()})
        print(json.dumps({'seed': seed, 'object': task.object, 'contact_candidates': int(labels.sum()),
                          'chosen_contact': bool(labels[selected]), 'unchanged_contact': bool(labels[0])}), flush=True)

    if len(scenarios) != ranking['episodes']:
        raise ValueError(f'Expected {ranking["episodes"]} ranking scenes; recorded {len(scenarios)}.')
    result = {'protocol': protocol, 'scenes': len(scenarios),
              'candidate_branches': sum(len(row['actual_contact']) for row in scenarios),
              'contact_head_top1_contact_rate': float(np.mean([row['contact_head_top1_contact'] for row in scenarios])),
              'old_held_top1_contact_rate': float(np.mean([row['old_held_top1_contact'] for row in scenarios])),
              'unchanged_top1_contact_rate': float(np.mean([row['unchanged_top1_contact'] for row in scenarios])),
              'oracle_contact_exists_rate': float(np.mean([row['oracle_has_contact'] for row in scenarios])),
              'contact_head_macro_average_precision': float(np.nanmean([row['contact_head_average_precision'] for row in scenarios])),
              'old_held_macro_average_precision': float(np.nanmean([row['old_held_average_precision'] for row in scenarios])),
              'per_object': {}, 'rows': scenarios,
              'interpretation': 'Offline ranking check on 40 new scenes. This did not run or change the controller.'}
    for name in OBJECTS:
        rows = [row for row in scenarios if row['object'] == name]
        result['per_object'][name] = {'scenes': len(rows),
                                      'head_top1_contact_rate': float(np.mean([x['contact_head_top1_contact'] for x in rows])),
                                      'old_held_top1_contact_rate': float(np.mean([x['old_held_top1_contact'] for x in rows])),
                                      'unchanged_contact_rate': float(np.mean([x['unchanged_top1_contact'] for x in rows])),
                                      'oracle_contact_exists_rate': float(np.mean([x['oracle_has_contact'] for x in rows])),
                                      'head_average_precision': float(np.nanmean([x['contact_head_average_precision'] for x in rows]))}
    (root / 'candidate_rank.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k not in ('rows', 'protocol')}, indent=2), flush=True)


if __name__ == '__main__':
    main()

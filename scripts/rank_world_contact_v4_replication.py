"""Run a frozen, fresh-seed replication of the version 4 ranking audit."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pybullet as p
import torch

from tinyvla import world_contact_targeted as targeted
from tinyvla.evaluate import load_policy
from tinyvla.language import OBJECTS, parse
from tinyvla.sim import ArmSim
from tinyvla.world_contact_rank import average_precision, candidates
from tinyvla.world_contact_train import ContactHead
from tinyvla.world_data import STATE_OFFSET, STATE_SCALE, tip_object
from tinyvla.world_model import load_world


ROOT = Path('artifacts/world-v4/contact')
SEED_START = 187020
SCENES = 40


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    torch.set_num_threads(1)
    config = json.loads(Path('configs/world_contact_v4.json').read_text())
    config['data']['ranking_seed'] = SEED_START
    config['data']['ranking_episodes'] = SCENES
    replication_root = ROOT / 'replication'
    protocol_path = ROOT / 'ranking_replication_protocol.json'
    result_path = ROOT / 'ranking_replication_results.json'
    if protocol_path.exists() or result_path.exists():
        raise FileExistsError('The replication protocol already exists. Preserve this result and use new seeds.')
    model_path = Path(config['source']['checkpoint'])
    head_path = ROOT / 'contact-head.pt'
    hashes = targeted.source_hashes()
    hashes['scripts/rank_world_contact_v4_replication.py'] = sha('scripts/rank_world_contact_v4_replication.py')
    hashes[str(head_path)] = sha(head_path)
    protocol = {'seed_start': SEED_START, 'seed_end': SEED_START + SCENES - 1,
                'scenes': SCENES, 'candidates_per_scene': 25,
                'object': 'red cylinder', 'contact_threshold_n': .01,
                'independent_replication_of': '187000-187019',
                'model_hashes': hashes, 'training_report_sha256': sha(ROOT / 'report.json'),
                'warning': 'Offline candidate ranking only. No controller test.'}
    protocol_path.write_text(json.dumps(protocol, indent=2))

    # Use the frozen version 4 collector with a new output folder and seed range.
    targeted.ROOT = replication_root
    targeted.collect_targeted(config, 'ranking')
    base, _ = load_world(model_path)
    bundle = torch.load(head_path, map_location='cpu', weights_only=False)
    head = ContactHead(bundle['latent_size'], bundle['hidden_size'])
    head.load_state_dict(bundle['state_dict']); head.eval()
    scale = torch.tensor(bundle['calibration_scale'])
    bias = torch.tensor(bundle['calibration_bias'])
    rows = []
    folder = replication_root / 'ranking'
    for path in sorted(folder.glob('*.npz')):
        with np.load(path) as data:
            features = data['features'].astype(np.float32)
            actions = data['actions'].astype(np.float32)
            seed = int(data['seed'])
        metadata = json.loads(path.with_suffix('.json').read_text())
        task = parse(metadata['instruction'])
        object_index = OBJECTS.index(task.object)
        choice = next((i for i in range(1, len(actions) - 3)
                       if actions[i, 2] <= .055 and actions[i, 3] > .02), None)
        if choice is None:
            raise ValueError(f'No open-gripper approach state in scene {seed}.')
        history_start = max(0, choice - 2)
        frames = np.concatenate([np.repeat(features[:1], 2 - (choice - history_start), axis=0),
                                 features[history_start:choice + 1]], axis=0)
        xyz = candidates(actions[choice, :3].copy())
        sequence = np.zeros((len(xyz), 3, 5), np.float32)
        sequence[:, 0, :3], sequence[:, 0, 3:] = xyz, [.04, 150 / 240]
        sequence[:, 1, :3], sequence[:, 1, 3:] = xyz, [0, 190 / 240]
        sequence[:, 2, :3], sequence[:, 2, 3:] = xyz + [0, 0, .03], [0, 24 / 240]
        with torch.no_grad():
            latent = base.encode(torch.from_numpy(frames)[None]).expand(len(xyz), -1)
            states, outcomes = base.rollout(latent, torch.from_numpy(sequence))
            logits = head(states.permute(1, 0, 2, 3)).permute(1, 0, 2, 3)
            scores = (logits.mean(0)[:, 1, object_index] * scale[object_index] + bias[object_index]).sigmoid().numpy()
            old_scores = outcomes[:, :, 1, object_index * 8 + 7].mean(0).numpy().clip(0, 1)
        sim = ArmSim()
        actual = []
        try:
            sim.reset(seed)
            if metadata['tipped']:
                tip_object(sim, task)
            for action in actions[:choice]:
                sim.move(action[:3], action[3], round(float(action[4]) * 240))
            replay_error = float(np.max(np.abs((sim.state() - STATE_OFFSET) / STATE_SCALE - features[choice, -4:])))
            if replay_error > .005:
                raise ValueError(f'Replay error {replay_error:.5f} in scene {seed}.')
            state = p.saveState()
            ticks, commands, heights = sim.ticks, sim.command_count, sim.max_heights.copy()
            for branch in sequence:
                p.restoreState(state)
                sim.ticks, sim.command_count, sim.max_heights = ticks, commands, heights.copy()
                for step, action in enumerate(branch):
                    sim.move(action[:3], action[3], round(float(action[4]) * 240))
                    if step == 1:
                        break
                force = sum(max(0., float(point[9])) for point in p.getContactPoints(
                    bodyA=sim.arm, bodyB=sim.objects[task.object]) if point[3] in (9, 10))
                actual.append(force > .01)
            p.removeState(state)
        finally:
            sim.close()
        actual = np.asarray(actual, bool)
        head_index, old_index = int(scores.argmax()), int(old_scores.argmax())
        row = {'seed': seed, 'tipped': metadata['tipped'], 'replay_error': replay_error,
               'positive_candidates': int(actual.sum()),
               'head_top1_contact': bool(actual[head_index]), 'unchanged_contact': bool(actual[0]),
               'old_top1_contact': bool(actual[old_index]), 'oracle_contact': bool(actual.any()),
               'head_ap': average_precision(scores, actual), 'old_ap': average_precision(old_scores, actual),
               'actual_contact': actual.tolist(), 'head_scores': scores.tolist()}
        rows.append(row)
        print(json.dumps({'seed': seed, 'tipped': row['tipped'], 'contacts': row['positive_candidates'],
                          'head_contact': row['head_top1_contact'],
                          'unchanged_contact': row['unchanged_contact']}), flush=True)
    if len(rows) != SCENES:
        raise ValueError(f'Expected {SCENES} scenes; got {len(rows)}.')
    wins = sum(row['head_top1_contact'] and not row['unchanged_contact'] for row in rows)
    losses = sum(row['unchanged_contact'] and not row['head_top1_contact'] for row in rows)
    result = {'protocol': protocol, 'scenes': len(rows), 'branches': len(rows) * 25,
              'head_top1_rate': float(np.mean([row['head_top1_contact'] for row in rows])),
              'unchanged_rate': float(np.mean([row['unchanged_contact'] for row in rows])),
              'old_top1_rate': float(np.mean([row['old_top1_contact'] for row in rows])),
              'oracle_rate': float(np.mean([row['oracle_contact'] for row in rows])),
              'head_mean_ap': float(np.nanmean([row['head_ap'] for row in rows])),
              'old_mean_ap': float(np.nanmean([row['old_ap'] for row in rows])),
              'paired_wins_vs_unchanged': wins, 'paired_losses_vs_unchanged': losses,
              'rows': rows,
              'interpretation': 'Independent, fresh-seed red-cylinder contact-ranking replication. No controller test.'}
    result_path.write_text(json.dumps(result, indent=2))
    print(json.dumps({key: value for key, value in result.items() if key not in ('rows', 'protocol')}, indent=2))


if __name__ == '__main__':
    main()

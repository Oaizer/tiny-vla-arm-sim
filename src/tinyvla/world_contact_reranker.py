"""Train and audit a small offline grasp-candidate reranker."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pybullet as p
import torch
from torch import nn

from . import world_contact_targeted as targeted
from .evaluate import load_policy
from .language import OBJECTS, parse
from .sim import ArmSim
from .world_contact_rank import average_precision, candidates
from .world_contact_train import ContactHead
from .world_data import STATE_OFFSET, STATE_SCALE, tip_object
from .world_model import load_world

ROOT = Path('artifacts/world-v5/reranker')
CONFIG_PATH = Path('configs/world_contact_ranker_v1.json')
FEATURE_NAMES = ['contact_probability_1', 'contact_probability_2', 'contact_probability_3',
                 'old_held_score_1', 'old_held_score_2', 'old_held_score_3',
                 'candidate_dx', 'candidate_dy', 'candidate_dz',
                 'target_dx', 'target_dy', 'target_dz', 'object_tipped']


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def config():
    return json.loads(CONFIG_PATH.read_text())


def split_config(cfg, split):
    value = json.loads(Path('configs/world_contact_v4.json').read_text())
    data = value['data']
    source = cfg['data']
    data[f'{split}_episodes'] = source[f'{split}_episodes']
    data[f'{split}_seed'] = source[f'{split}_seed']
    data['tip_probability'] = source['tip_probability']
    data['drop_probability'] = source['drop_probability']
    data['high_noise_probability'] = source['high_noise_probability']
    data['high_noise_std'] = source['high_noise_std']
    data['low_noise_std'] = source['low_noise_std']
    value['source']['policy'] = cfg['source']['policy']
    return value


def collect(split=None):
    cfg = config()
    targeted.ROOT = ROOT / 'episodes'
    splits = [split] if split else ['train', 'validation', 'test']
    for name in splits:
        targeted.collect_targeted(split_config(cfg, name), name)


def source_hashes():
    paths = [CONFIG_PATH, Path('configs/world_contact_v4.json'), Path('src/tinyvla/world_contact_reranker.py'),
             Path('src/tinyvla/world_contact_targeted.py'), Path('src/tinyvla/world_contact_data.py'),
             Path('src/tinyvla/world_contact_train.py'), Path('src/tinyvla/world_contact_rank.py'),
             Path('src/tinyvla/world_model.py'), Path('src/tinyvla/world_data.py'), Path('src/tinyvla/sim.py'),
             Path('src/tinyvla/evaluate.py'), Path('src/tinyvla/language.py'), Path('src/tinyvla/controller.py'),
             Path('configs/world_contact.json'), Path('uv.lock'), Path('artifacts/world-v1/model/jepa.pt'),
             Path('artifacts/world-v4/contact/contact-head.pt'), Path('artifacts/model/best.pt')]
    return {str(path): digest(path) for path in paths}


def selected_command(actions):
    choice = next((i for i in range(1, len(actions) - 3)
                   if actions[i, 2] <= .055 and actions[i, 3] > .02), None)
    if choice is None:
        raise ValueError('No open-gripper candidate point found.')
    return choice


def build():
    ROOT.mkdir(parents=True, exist_ok=True)
    protocol_path = ROOT / 'protocol.json'
    if protocol_path.exists() or any((ROOT / f'branches_{s}.npz').exists() for s in ('train', 'validation', 'test')):
        raise FileExistsError('A version 5 branch build already exists. Preserve it and use new seeds for a new run.')
    cfg = config()
    episodes = {}
    for split in ('train', 'validation', 'test'):
        folder = ROOT / 'episodes' / split
        files = sorted(folder.glob('*.npz'))
        expected = cfg['data'][f'{split}_episodes']
        if len(files) != expected:
            raise ValueError(f'{split}: expected {expected} episodes, found {len(files)}.')
        episodes[split] = {x.name: digest(x) for x in files}
    base, _ = load_world(cfg['source']['world_model'])
    bundle = torch.load(cfg['source']['contact_head'], map_location='cpu', weights_only=False)
    head = ContactHead(bundle['latent_size'], bundle['hidden_size'])
    head.load_state_dict(bundle['state_dict']); head.eval()
    protocol = {'version': 1, 'created_unix': int(time.time()), 'seeds': {
        split: list(range(cfg['data'][f'{split}_seed'], cfg['data'][f'{split}_seed'] + cfg['data'][f'{split}_episodes']))
        for split in ('train', 'validation', 'test')}, 'candidate_count': 25,
        'decision': 'first recorded open-gripper command after the initial approach',
        'candidate_execution': 'offline branch from a saved PyBullet state; execute approach and close only',
        'contact_threshold_n': .01, 'features': FEATURE_NAMES,
        'source_hashes': source_hashes(), 'episode_sha256': episodes,
        'contact_head_sha256': digest(Path(cfg['source']['contact_head']))}
    protocol_path.write_text(json.dumps(protocol, indent=2))
    for split in ('train', 'validation', 'test'):
        rows_x, rows_y, rows_seed = [], [], []
        for path in sorted((ROOT / 'episodes' / split).glob('*.npz')):
            with np.load(path) as record:
                features = record['features'].astype(np.float32)
                actions = record['actions'].astype(np.float32)
                seed = int(record['seed'])
            metadata = json.loads(path.with_suffix('.json').read_text())
            task = parse(metadata['instruction']); source = OBJECTS.index(task.object)
            choice = selected_command(actions)
            history_start = max(0, choice - 2)
            history_frames = np.concatenate([np.repeat(features[:1], 2 - (choice - history_start), axis=0),
                                             features[history_start:choice + 1]], axis=0)
            xyz = candidates(actions[choice, :3].copy())
            sequence = np.zeros((len(xyz), 3, 5), np.float32)
            sequence[:, 0, :3], sequence[:, 0, 3:] = xyz, [.04, 150 / 240]
            sequence[:, 1, :3], sequence[:, 1, 3:] = xyz, [0, 190 / 240]
            sequence[:, 2, :3], sequence[:, 2, 3:] = xyz + [0, 0, .03], [0, 24 / 240]
            with torch.no_grad():
                latent = base.encode(torch.from_numpy(history_frames)[None]).expand(len(xyz), -1)
                states, outcomes = base.rollout(latent, torch.from_numpy(sequence))
                logits = head(states.permute(1, 0, 2, 3)).permute(1, 0, 2, 3)
                probabilities = logits.sigmoid()[:, :, 1, source].T.numpy()
                old_scores = outcomes[:, :, 1, source * 8 + 7].T.numpy().clip(0, 1)
                predicted_xyz = outcomes[:, :, 1, source * 8:source * 8 + 3].mean(0).numpy()
                predicted_xyz = predicted_xyz * STATE_SCALE[:3] + STATE_OFFSET[:3]
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
                    raise ValueError(f'Replay error {replay_error:.5f} in seed {seed}.')
                state_id = p.saveState(); ticks, count = sim.ticks, sim.command_count
                max_heights = sim.max_heights.copy()
                try:
                    for candidate in sequence:
                        p.restoreState(state_id); sim.ticks, sim.command_count = ticks, count
                        sim.max_heights = max_heights.copy()
                        for action_index, action in enumerate(candidate):
                            sim.move(action[:3], action[3], round(float(action[4]) * 240))
                            if action_index == 1:
                                break
                        force = sum(max(0., float(point[9])) for point in p.getContactPoints(
                            bodyA=sim.arm, bodyB=sim.objects[task.object]) if point[3] in (9, 10))
                        actual.append(force > .01)
                finally:
                    p.removeState(state_id)
            finally:
                sim.close()
            labels = np.asarray(actual, dtype=np.float32)
            candidate_features = np.concatenate([probabilities, old_scores, xyz - actions[choice, :3],
                                                  xyz - predicted_xyz, np.full((len(xyz), 1), float(metadata['tipped']))], axis=1)
            rows_x.append(candidate_features.astype(np.float32)); rows_y.append(labels)
            rows_seed.append(np.full(len(xyz), seed, np.int64))
            print(json.dumps({'split': split, 'seed': seed, 'positive_candidates': int(labels.sum()),
                              'replay_error': replay_error}), flush=True)
        np.savez_compressed(ROOT / f'branches_{split}.npz',
                            features=np.concatenate(rows_x), labels=np.concatenate(rows_y),
                            scene_seed=np.concatenate(rows_seed))
    protocol['branch_sha256'] = {f'{s}': digest(ROOT / f'branches_{s}.npz') for s in ('train', 'validation', 'test')}
    protocol_path.write_text(json.dumps(protocol, indent=2))


def scene_metrics(scores, labels, seeds):
    aps, top = [], []
    for seed in np.unique(seeds):
        mask = seeds == seed
        y, score = labels[mask], scores[mask]
        if y.any():
            aps.append(average_precision(score, y))
        top.append(bool(y[int(score.argmax())]))
    return {'scenes': len(top), 'scenes_with_positive': len(aps),
            'macro_average_precision': float(np.mean(aps)) if aps else float('nan'),
            'top1_contact_rate': float(np.mean(top))}


class Ranker(nn.Module):
    def __init__(self, n_features, hidden):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(n_features, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, value):
        return self.layers(value).squeeze(-1)


def train():
    cfg = config(); settings = cfg['training']
    torch.manual_seed(settings['seed']); np.random.seed(settings['seed'])
    torch.use_deterministic_algorithms(True); torch.set_num_threads(1)
    with np.load(ROOT / 'branches_train.npz') as data:
        x_train, y_train = data['features'], data['labels']
    with np.load(ROOT / 'branches_validation.npz') as data:
        x_val, y_val, seed_val = data['features'], data['labels'], data['scene_seed']
    mean, std = x_train.mean(0), x_train.std(0).clip(1e-5)
    tx = torch.from_numpy((x_train - mean) / std); ty = torch.from_numpy(y_train)
    vx = torch.from_numpy((x_val - mean) / std)
    model = Ranker(x_train.shape[1], settings['hidden_size'])
    positive = max(float(y_train.sum()), 1.)
    negative = max(float(len(y_train) - y_train.sum()), 1.)
    weight = torch.tensor(min(settings['positive_weight_max'], max(1., negative / positive)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'])
    log_path = ROOT / 'training.jsonl'; log_path.write_text('')
    best_ap, best_step, best_state = -1., 0, None
    started = time.time()
    for step in range(1, settings['steps'] + 1):
        model.train(); logits = model(tx)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, ty, pos_weight=weight)
        optimizer.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5); optimizer.step()
        if step == 1 or step == settings['steps'] or step % 10 == 0:
            model.eval()
            with torch.no_grad():
                scores = model(vx).numpy()
            metric = scene_metrics(scores, y_val, seed_val)
            entry = {'step': step, 'loss': float(loss.detach()), 'validation': metric,
                     'seconds': round(time.time() - started, 1)}
            with log_path.open('a') as stream:
                stream.write(json.dumps(entry) + '\n')
            if np.isfinite(metric['macro_average_precision']) and metric['macro_average_precision'] > best_ap:
                best_ap, best_step = metric['macro_average_precision'], step
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            if step % 100 == 0 or step == settings['steps']:
                print(json.dumps({'step': step, 'validation': metric}), flush=True)
    if best_state is None:
        raise ValueError('Validation set has no positive contact scenes.')
    model.load_state_dict(best_state)
    bundle = {'state_dict': model.state_dict(), 'mean': mean, 'std': std,
              'feature_names': FEATURE_NAMES, 'hidden_size': settings['hidden_size'],
              'protocol_sha256': digest(ROOT / 'protocol.json'), 'selected_step': best_step}
    torch.save(bundle, ROOT / 'ranker.pt')
    report = {'selected_step': best_step, 'validation': scene_metrics(model(torch.from_numpy((x_val - mean) / std)).detach().numpy(), y_val, seed_val),
              'training_scenes': len(np.unique(np.load(ROOT / 'branches_train.npz')['scene_seed'])),
              'parameters': sum(p.numel() for p in model.parameters()),
              'protocol_sha256': bundle['protocol_sha256'], 'ranker_sha256': digest(ROOT / 'ranker.pt')}
    (ROOT / 'ranker_report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


def evaluate():
    cfg = config(); torch.set_num_threads(1)
    bundle = torch.load(ROOT / 'ranker.pt', map_location='cpu', weights_only=False)
    model = Ranker(len(bundle['feature_names']), bundle['hidden_size'])
    model.load_state_dict(bundle['state_dict']); model.eval()
    with np.load(ROOT / 'branches_test.npz') as data:
        x, y, seeds = data['features'], data['labels'], data['scene_seed']
    with torch.no_grad():
        learned = model(torch.from_numpy((x - bundle['mean']) / bundle['std'])).numpy()
    metrics = {'reranker': scene_metrics(learned, y, seeds),
               'contact_head': scene_metrics(x[:, :3].mean(1), y, seeds),
               'old_geometric_score': scene_metrics(x[:, 3:6].mean(1), y, seeds),
               'unchanged_command': scene_metrics(-np.tile(np.arange(25), len(y) // 25), y, seeds)}
    wins = {'vs_unchanged': 0, 'ties_vs_unchanged': 0, 'losses_vs_unchanged': 0}
    pairs = []
    for seed in np.unique(seeds):
        mask = seeds == seed; labels = y[mask]
        ranker_hit = bool(labels[int(learned[mask].argmax())])
        unchanged_hit = bool(labels[0])
        pairs.append((ranker_hit, unchanged_hit))
        if ranker_hit and not unchanged_hit: wins['vs_unchanged'] += 1
        elif ranker_hit == unchanged_hit: wins['ties_vs_unchanged'] += 1
        else: wins['losses_vs_unchanged'] += 1
    protocol = json.loads((ROOT / 'protocol.json').read_text())
    result = {'protocol_sha256': digest(ROOT / 'protocol.json'), 'ranker_sha256': digest(ROOT / 'ranker.pt'),
              'test_metrics': metrics, 'paired_top1_vs_unchanged': wins,
              'test_scenes': len(np.unique(seeds)), 'test_branches': len(y),
              'interpretation': 'Offline red-cylinder contact ranking only. The reranker did not control the arm.'}
    (ROOT / 'test_results.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['collect', 'build', 'train', 'evaluate'])
    parser.add_argument('--split', choices=['train', 'validation', 'test'])
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    if args.action == 'collect': collect(args.split)
    elif args.action == 'build': build()
    elif args.action == 'train': train()
    else: evaluate()


if __name__ == '__main__':
    main()

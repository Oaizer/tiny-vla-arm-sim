"""Train a corrected red-cylinder reranker and test it on fresh branches."""
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
from .language import OBJECTS, parse
from .sim import ArmSim
from .world_contact_rank import candidates, average_precision
from .world_contact_reranker import Ranker, scene_metrics
from .world_contact_train import ContactHead, OBJECTS as CONTACT_OBJECTS
from .world_data import STATE_OFFSET, STATE_SCALE, tip_object
from .world_model import load_world

ROOT = Path('artifacts/world-v7/corrected-ranker')
V5_ROOT = Path('artifacts/world-v5/reranker')
CONFIG_PATH = Path('configs/world_hybrid_v7.json')
CONTACT_CHANNEL = CONTACT_OBJECTS.index('red cylinder')
STATE_CHANNEL = OBJECTS.index('red cylinder')
FEATURE_NAMES = ['contact_probability_1', 'contact_probability_2', 'contact_probability_3',
                 'old_held_score_1', 'old_held_score_2', 'old_held_score_3',
                 'candidate_dx', 'candidate_dy', 'candidate_dz',
                 'target_dx', 'target_dy', 'target_dz', 'unavailable_tipped_feature_zero']


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_inputs(episodes, labels_by_seed, base, head):
    rows_x, rows_y, rows_seed, audits = [], [], [], []
    for path in sorted(episodes.glob('*.npz'), key=lambda item: int(item.stem)):
        with np.load(path) as stored:
            features = stored['features'].astype(np.float32)
            actions = stored['actions'].astype(np.float32)
            seed = int(stored['seed'])
        metadata = json.loads(path.with_suffix('.json').read_text())
        task = parse(metadata['instruction'])
        choice = next((i for i in range(1, len(actions) - 3)
                       if actions[i, 2] <= .055 and actions[i, 3] > .02), None)
        if choice is None:
            raise ValueError(f'No open-gripper grasp point in seed {seed}.')
        history_start = max(0, choice - 2)
        history = np.concatenate([np.repeat(features[:1], 2 - (choice - history_start), axis=0),
                                  features[history_start:choice + 1]], axis=0)
        points = candidates(actions[choice, :3].copy())
        sequence = np.zeros((25, 3, 5), np.float32)
        sequence[:, 0, :3], sequence[:, 0, 3:] = points, [.04, 150 / 240]
        sequence[:, 1, :3], sequence[:, 1, 3:] = points, [0, 190 / 240]
        sequence[:, 2, :3], sequence[:, 2, 3:] = points + [0, 0, .03], [0, 24 / 240]
        with torch.inference_mode():
            latent = base.encode(torch.from_numpy(history)[None]).expand(25, -1)
            states, outcomes = base.rollout(latent, torch.from_numpy(sequence))
            logits = head(states.permute(1, 0, 2, 3)).permute(1, 0, 2, 3)
            # ContactHead and language state use different channel orders.
            # Use the explicitly named red-cylinder channel for this feature.
            contact = logits.sigmoid()[:, :, 1, CONTACT_CHANNEL].T.numpy()
            held = outcomes[:, :, 1, STATE_CHANNEL * 8 + 7].T.numpy().clip(0, 1)
            target = outcomes[:, :, 1, STATE_CHANNEL * 8:STATE_CHANNEL * 8 + 3].mean(0).numpy()
            target = target * STATE_SCALE[:3] + STATE_OFFSET[:3]
        geometry = np.concatenate([points - actions[choice, :3], points - target,
                                   np.zeros((25, 1), np.float32)], axis=1)
        x = np.concatenate([contact, held, geometry], axis=1).astype(np.float32)
        if seed not in labels_by_seed:
            raise ValueError(f'No frozen contact labels found for seed {seed}.')
        y = labels_by_seed[seed]
        if x.shape != (25, 13) or y.shape != (25,):
            raise ValueError(f'Unexpected branch data shape for seed {seed}.')
        rows_x.append(x); rows_y.append(y); rows_seed.append(np.full(25, seed, np.int64))
        audits.append({'seed': seed, 'object': task.object, 'tipped_training_label_omitted': True,
                       'contact_channel': CONTACT_CHANNEL, 'state_channel': STATE_CHANNEL})
    return np.concatenate(rows_x), np.concatenate(rows_y), np.concatenate(rows_seed), audits


def labels_by_seed(branch_path):
    with np.load(branch_path) as data:
        y, seeds = data['labels'], data['scene_seed']
    return {int(seed): y[seeds == seed].astype(np.float32) for seed in np.unique(seeds)}


def collect_fresh(cfg):
    v4 = json.loads(Path('configs/world_contact_v4.json').read_text())
    spec = cfg['ranking']
    v4['data']['test_episodes'] = spec['episodes']
    v4['data']['test_seed'] = spec['seed_start']
    v4['data']['tip_probability'] = spec['tip_probability']
    v4['data']['drop_probability'] = spec['drop_probability']
    v4['data']['high_noise_probability'] = spec['high_noise_probability']
    v4['data']['high_noise_std'] = spec['high_noise_std']
    v4['data']['low_noise_std'] = spec['low_noise_std']
    targeted.ROOT = ROOT / 'episodes'
    targeted.collect_targeted(v4, 'test')


def collect_labels(paths, cfg):
    spec = cfg['ranking']; labels_by_scene = {}; replay_checks = {}
    sim = ArmSim()
    try:
        for path in sorted(paths.glob('*.npz'), key=lambda item: int(item.stem)):
            with np.load(path) as stored:
                features = stored['features'].astype(np.float32)
                actions = stored['actions'].astype(np.float32)
                seed = int(stored['seed'])
            metadata = json.loads(path.with_suffix('.json').read_text())
            task = parse(metadata['instruction'])
            choice = next((i for i in range(1, len(actions) - 3)
                           if actions[i, 2] <= .055 and actions[i, 3] > .02), None)
            if choice is None: raise ValueError(f'No open-gripper grasp point in seed {seed}.')
            xyz = candidates(actions[choice, :3].copy())
            sequence = np.zeros((25, 3, 5), np.float32)
            sequence[:, 0, :3], sequence[:, 0, 3:] = xyz, [.04, 150 / 240]
            sequence[:, 1, :3], sequence[:, 1, 3:] = xyz, [0, 190 / 240]
            sequence[:, 2, :3], sequence[:, 2, 3:] = xyz + [0, 0, .03], [0, 24 / 240]
            sim.reset(seed)
            if metadata['tipped']: tip_object(sim, task)
            for action in actions[:choice]:
                sim.move(action[:3], action[3], round(float(action[4]) * 240))
            state_error = float(np.max(np.abs((sim.state() - STATE_OFFSET) / STATE_SCALE - features[choice, -4:])))
            if state_error > .005: raise ValueError(f'Replay error {state_error:.5f} for seed {seed}.')
            state_id = p.saveState(); ticks, command_count = sim.ticks, sim.command_count
            heights = sim.max_heights.copy(); labels = []
            try:
                for branch in sequence:
                    p.restoreState(state_id)
                    sim.ticks, sim.command_count = ticks, command_count
                    sim.max_heights = heights.copy()
                    for index, action in enumerate(branch):
                        sim.move(action[:3], action[3], round(float(action[4]) * 240))
                        if index == 1: break
                    force = sum(max(0., float(point[9])) for point in p.getContactPoints(
                        bodyA=sim.arm, bodyB=sim.objects[task.object]) if point[3] in (9, 10))
                    labels.append(force > spec['contact_threshold_n'])
            finally:
                p.removeState(state_id)
            labels_by_scene[seed] = np.asarray(labels, np.float32)
            replay_checks[seed] = {'state_error_max': state_error,
                                   'positive_candidates': int(np.sum(labels_by_scene[seed]))}
            print(json.dumps({'seed': seed, **replay_checks[seed]}), flush=True)
    finally:
        sim.close()
    return labels_by_scene, replay_checks


def train_and_test():
    cfg = json.loads(CONFIG_PATH.read_text())
    ROOT.mkdir(parents=True, exist_ok=True)
    protocol_path = ROOT / 'protocol.json'
    if protocol_path.exists(): raise FileExistsError('Preserve this V7 protocol and use a new output directory.')
    protocol_files = [CONFIG_PATH, Path('configs/world_contact_v4.json'),
                      Path('src/tinyvla/world_hybrid_ranker_v7.py'), Path('src/tinyvla/world_hybrid_v7_experiment.py'),
                      Path('src/tinyvla/world_hybrid_corrected.py'), Path('src/tinyvla/world_contact_targeted.py'),
                      Path('src/tinyvla/world_contact_data.py'), Path('src/tinyvla/world_contact_train.py'),
                      Path('src/tinyvla/world_contact_rank.py'), Path('src/tinyvla/world_contact_reranker.py'),
                      Path('src/tinyvla/world_model.py'), Path('src/tinyvla/world_data.py'),
                      Path('src/tinyvla/sim.py'), Path('src/tinyvla/evaluate.py'), Path('src/tinyvla/controller.py'),
                      Path('src/tinyvla/world_evaluate.py'), Path('src/tinyvla/language.py'), Path('uv.lock'),
                      Path('artifacts/world-v1/model/jepa.pt'), Path('artifacts/world-v4/contact/contact-head.pt'),
                      Path('artifacts/model/best.pt'), Path('artifacts/world-v5/reranker/ranker.pt'),
                      Path('artifacts/world-v5/reranker/protocol.json')]
    protocol = {'version': 1, 'created_unix': int(time.time()),
                'seed_start': cfg['ranking']['seed_start'], 'ranking_scenes': cfg['ranking']['episodes'],
                'candidate_branches_per_scene': 25, 'contact_threshold_n': cfg['ranking']['contact_threshold_n'],
                'training_scenes': {'train': 40, 'validation': 20},
                'training_inputs': 'V5 train and validation scenes with red-cylinder contact-head output channel 3. Simulator tipped labels are excluded from the ranker input.',
                'test_inputs': 'Fresh red-cylinder scenes collected and labelled by PyBullet saved-state branches.',
                'features': FEATURE_NAMES, 'source_sha256': {str(path): digest(path) for path in protocol_files}}
    protocol_path.write_text(json.dumps(protocol, indent=2) + '\n')

    base, _ = load_world('artifacts/world-v1/model/jepa.pt')
    bundle = torch.load('artifacts/world-v4/contact/contact-head.pt', map_location='cpu', weights_only=False)
    head = ContactHead(bundle['latent_size'], bundle['hidden_size'])
    head.load_state_dict(bundle['state_dict']); head.eval()
    split_results = {}
    for split in ('train', 'validation'):
        x, y, seeds, audits = build_inputs(V5_ROOT / 'episodes' / split,
                                           labels_by_seed(V5_ROOT / f'branches_{split}.npz'), base, head)
        np.savez_compressed(ROOT / f'branches_{split}.npz', features=x, labels=y, scene_seed=seeds)
        split_results[split] = {'scenes': len(np.unique(seeds)), 'branches': len(y), 'feature_audit': audits}

    collect_fresh(cfg)
    labels, replay_checks = collect_labels(ROOT / 'episodes' / 'test', cfg)
    x_test, y_test, seed_test, test_audits = build_inputs(ROOT / 'episodes' / 'test', labels, base, head)
    np.savez_compressed(ROOT / 'branches_test.npz', features=x_test, labels=y_test, scene_seed=seed_test)
    split_results['test'] = {'scenes': len(np.unique(seed_test)), 'branches': len(y_test),
                             'feature_audit': test_audits, 'replay_checks': replay_checks}

    settings = cfg['training']
    torch.manual_seed(settings['seed']); np.random.seed(settings['seed'])
    torch.use_deterministic_algorithms(True); torch.set_num_threads(1)
    with np.load(ROOT / 'branches_train.npz') as data: x_train, y_train = data['features'], data['labels']
    with np.load(ROOT / 'branches_validation.npz') as data: x_val, y_val, seed_val = data['features'], data['labels'], data['scene_seed']
    mean, std = x_train.mean(0), x_train.std(0).clip(1e-5)
    tx = torch.from_numpy((x_train - mean) / std); ty = torch.from_numpy(y_train)
    vx = torch.from_numpy((x_val - mean) / std)
    model = Ranker(x_train.shape[1], settings['hidden_size'])
    positives = max(float(y_train.sum()), 1.); negatives = max(float(len(y_train) - y_train.sum()), 1.)
    pos_weight = torch.tensor(min(settings['positive_weight_max'], max(1., negatives / positives)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'])
    best_ap, best_state, best_step = -1., None, 0
    training_log = []
    for step in range(1, settings['steps'] + 1):
        model.train(); logits = model(tx)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, ty, pos_weight=pos_weight)
        optimizer.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5); optimizer.step()
        if step == 1 or step % 10 == 0 or step == settings['steps']:
            model.eval()
            with torch.no_grad(): scores = model(vx).numpy()
            metric = scene_metrics(scores, y_val, seed_val)
            training_log.append({'step': step, 'loss': float(loss.detach()), 'validation': metric})
            if np.isfinite(metric['macro_average_precision']) and metric['macro_average_precision'] > best_ap:
                best_ap, best_step = metric['macro_average_precision'], step
                best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    if best_state is None: raise ValueError('V7 validation scenes contain no positive candidates.')
    model.load_state_dict(best_state); model.eval()
    torch.save({'state_dict': model.state_dict(), 'mean': mean, 'std': std,
                'feature_names': FEATURE_NAMES, 'hidden_size': settings['hidden_size'],
                'selected_step': best_step, 'protocol_sha256': digest(protocol_path)}, ROOT / 'ranker.pt')
    (ROOT / 'training.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in training_log))
    with np.load(ROOT / 'branches_test.npz') as data: x_test, y_test, seed_test = data['features'], data['labels'], data['scene_seed']
    with torch.no_grad(): scores = model(torch.from_numpy((x_test - mean) / std)).numpy()
    contact_score = x_test[:, :3].mean(1)
    geometric_score = x_test[:, 3:6].mean(1)
    unchanged_score = -np.tile(np.arange(25), len(y_test) // 25)
    result = {'protocol_sha256': digest(protocol_path), 'ranker_sha256': digest(ROOT / 'ranker.pt'),
              'split_scenes': {key: value['scenes'] for key, value in split_results.items()},
              'selected_step': best_step, 'parameters': sum(parameter.numel() for parameter in model.parameters()),
              'test_scenes': int(len(np.unique(seed_test))), 'test_branches': int(len(y_test)),
              'test_metrics': {'corrected_reranker': scene_metrics(scores, y_test, seed_test),
                               'contact_head_red_cylinder': scene_metrics(contact_score, y_test, seed_test),
                               'old_geometric_score': scene_metrics(geometric_score, y_test, seed_test),
                               'unchanged_command': scene_metrics(unchanged_score, y_test, seed_test)},
              'paired_vs_unchanged': {'wins': 0, 'ties': 0, 'losses': 0},
              'scene_rows': []}
    for seed in np.unique(seed_test):
        mask = seed_test == seed; labels = y_test[mask]
        top = int(scores[mask].argmax()); unchanged = bool(labels[0]); selected = bool(labels[top])
        if selected and not unchanged: result['paired_vs_unchanged']['wins'] += 1
        elif selected == unchanged: result['paired_vs_unchanged']['ties'] += 1
        else: result['paired_vs_unchanged']['losses'] += 1
        result['scene_rows'].append({'seed': int(seed), 'corrected_ranker_choice': top,
                                     'ranker_contact': selected, 'unchanged_contact': unchanged,
                                     'positive_candidates': int(labels.sum()),
                                     'average_precision': average_precision(scores[mask], labels)})
    result['interpretation'] = 'Fresh offline contact-ranking check for a corrected red-cylinder contact feature. Contact after closure is not task success.'
    (ROOT / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    manifest = {'protocol_sha256': digest(protocol_path),
                'dataset_sha256': {name: digest(ROOT / f'branches_{name}.npz') for name in ('train', 'validation', 'test')},
                'test_episode_sha256': {path.name: digest(path) for path in sorted((ROOT / 'episodes/test').glob('*.npz'))}}
    (ROOT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key not in ('scene_rows',)}, indent=2), flush=True)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['run'])
    parser.parse_args()
    train_and_test()


if __name__ == '__main__':
    main()

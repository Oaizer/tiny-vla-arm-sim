"""Run an independent fresh-scene audit of the frozen version 5 reranker."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pybullet as p
import torch

from . import world_contact_targeted as targeted
from .language import OBJECTS, parse
from .sim import ArmSim
from .world_contact_rank import candidates
from .world_contact_reranker import (FEATURE_NAMES, ROOT as V5_ROOT, Ranker, average_precision,
                                     digest, scene_metrics)
from .world_contact_train import ContactHead
from .world_data import STATE_OFFSET, STATE_SCALE, tip_object
from .world_model import load_world

ROOT = Path('artifacts/world-v5/replication')
CONFIG_PATH = Path('configs/world_contact_ranker_replication_v1.json')


def config():
    return json.loads(CONFIG_PATH.read_text())


def hashes():
    paths = [CONFIG_PATH, Path('src/tinyvla/world_contact_reranker_replication.py'),
             Path('src/tinyvla/world_contact_reranker.py'), Path('src/tinyvla/world_contact_targeted.py'),
             Path('src/tinyvla/world_contact_data.py'), Path('src/tinyvla/world_contact_train.py'),
             Path('src/tinyvla/world_contact_rank.py'), Path('src/tinyvla/world_model.py'),
             Path('src/tinyvla/world_data.py'), Path('src/tinyvla/sim.py'), Path('src/tinyvla/language.py'),
             Path('configs/world_contact_v4.json'), Path('uv.lock'),
             Path('artifacts/world-v1/model/jepa.pt'), Path('artifacts/world-v4/contact/contact-head.pt'),
             Path('artifacts/model/best.pt'), V5_ROOT / 'protocol.json',
             V5_ROOT / 'ranker.pt', V5_ROOT / 'ranker_report.json', V5_ROOT / 'test_results.json']
    return {str(path): digest(path) for path in paths}


def collect(cfg):
    value = json.loads(Path('configs/world_contact_v4.json').read_text())
    data = value['data']; spec = cfg['data']
    data['test_episodes'] = spec['episodes']; data['test_seed'] = spec['seed_start']
    for key in ('tip_probability', 'drop_probability', 'high_noise_probability', 'high_noise_std', 'low_noise_std'):
        data[key] = spec[key]
    targeted.ROOT = ROOT / 'episodes'
    targeted.collect_targeted(value, 'test')


def run():
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    cfg = config(); spec = cfg['data']
    ROOT.mkdir(parents=True, exist_ok=True)
    protocol_path = ROOT / 'protocol.json'
    if protocol_path.exists() or (ROOT / 'branches.npz').exists() or (ROOT / 'results.json').exists():
        raise FileExistsError('A replication protocol already exists. Preserve it and use new seeds for another study.')
    v5_protocol = json.loads((V5_ROOT / 'protocol.json').read_text())
    protocol = {'version': 1, 'created_unix': int(time.time()), 'seed_start': spec['seed_start'],
                'seed_end': spec['seed_start'] + spec['episodes'] - 1, 'scenes': spec['episodes'],
                'candidates_per_scene': 25, 'contact_threshold_n': spec['contact_threshold_n'],
                'frozen_v5_protocol_sha256': digest(V5_ROOT / 'protocol.json'),
                'v5_test_report_sha256': digest(Path(cfg['frozen_test_report'])),
                'source_hashes': hashes(), 'torch_threads': 1,
                'decision': 'first recorded open-gripper command after the initial approach',
                'branches': 'offline saved-state copies; close only; no controller use'}
    protocol_path.write_text(json.dumps(protocol, indent=2))
    collect(cfg)

    base, _ = load_world('artifacts/world-v1/model/jepa.pt')
    contact_bundle = torch.load('artifacts/world-v4/contact/contact-head.pt', map_location='cpu', weights_only=False)
    contact_head = ContactHead(contact_bundle['latent_size'], contact_bundle['hidden_size'])
    contact_head.load_state_dict(contact_bundle['state_dict']); contact_head.eval()
    rank_bundle = torch.load(V5_ROOT / 'ranker.pt', map_location='cpu', weights_only=False)
    ranker = Ranker(len(rank_bundle['feature_names']), rank_bundle['hidden_size'])
    ranker.load_state_dict(rank_bundle['state_dict']); ranker.eval()
    mean, std = rank_bundle['mean'], rank_bundle['std']

    all_features, all_labels, all_seeds = [], [], []
    episode_hashes = {}
    for path in sorted((ROOT / 'episodes' / 'test').glob('*.npz')):
        with np.load(path) as record:
            features = record['features'].astype(np.float32); actions = record['actions'].astype(np.float32)
            seed = int(record['seed'])
        meta = json.loads(path.with_suffix('.json').read_text())
        task = parse(meta['instruction']); source = OBJECTS.index(task.object)
        choice = next((i for i in range(1, len(actions) - 3)
                       if actions[i, 2] <= .055 and actions[i, 3] > .02), None)
        if choice is None: raise ValueError(f'No open-gripper candidate point in seed {seed}.')
        history_start = max(0, choice - 2)
        history = np.concatenate([np.repeat(features[:1], 2 - (choice - history_start), axis=0),
                                  features[history_start:choice + 1]], axis=0)
        xyz = candidates(actions[choice, :3].copy())
        sequence = np.zeros((25, 3, 5), np.float32)
        sequence[:, 0, :3], sequence[:, 0, 3:] = xyz, [.04, 150 / 240]
        sequence[:, 1, :3], sequence[:, 1, 3:] = xyz, [0, 190 / 240]
        sequence[:, 2, :3], sequence[:, 2, 3:] = xyz + [0, 0, .03], [0, 24 / 240]
        with torch.no_grad():
            latent = base.encode(torch.from_numpy(history)[None]).expand(25, -1)
            states, outcomes = base.rollout(latent, torch.from_numpy(sequence))
            logits = contact_head(states.permute(1, 0, 2, 3)).permute(1, 0, 2, 3)
            probs = logits.sigmoid()[:, :, 1, source].T.numpy()
            old_scores = outcomes[:, :, 1, source * 8 + 7].T.numpy().clip(0, 1)
            target_xyz = outcomes[:, :, 1, source * 8:source * 8 + 3].mean(0).numpy()
            target_xyz = target_xyz * STATE_SCALE[:3] + STATE_OFFSET[:3]
        sim = ArmSim(); labels = []
        try:
            sim.reset(seed)
            if meta['tipped']: tip_object(sim, task)
            for action in actions[:choice]: sim.move(action[:3], action[3], round(float(action[4]) * 240))
            replay_error = float(np.max(np.abs((sim.state() - STATE_OFFSET) / STATE_SCALE - features[choice, -4:])))
            if replay_error > .005: raise ValueError(f'Scene replay error {replay_error:.5f} in seed {seed}.')
            state_id = p.saveState(); ticks, count = sim.ticks, sim.command_count; heights = sim.max_heights.copy()
            try:
                for candidate in sequence:
                    p.restoreState(state_id); sim.ticks, sim.command_count = ticks, count; sim.max_heights = heights.copy()
                    for action_index, action in enumerate(candidate):
                        sim.move(action[:3], action[3], round(float(action[4]) * 240))
                        if action_index == 1: break
                    force = sum(max(0., float(point[9])) for point in p.getContactPoints(
                        bodyA=sim.arm, bodyB=sim.objects[task.object]) if point[3] in (9, 10))
                    labels.append(force > spec['contact_threshold_n'])
            finally: p.removeState(state_id)
        finally: sim.close()
        labels = np.asarray(labels, dtype=np.float32)
        fx = np.concatenate([probs, old_scores, xyz - actions[choice, :3], xyz - target_xyz,
                             np.full((25, 1), float(meta['tipped']))], axis=1).astype(np.float32)
        all_features.append(fx); all_labels.append(labels); all_seeds.append(np.full(25, seed, np.int64))
        episode_hashes[path.name] = digest(path)
        print(json.dumps({'seed': seed, 'positive_candidates': int(labels.sum()),
                          'replay_error': replay_error}), flush=True)
    x = np.concatenate(all_features); y = np.concatenate(all_labels); seeds = np.concatenate(all_seeds)
    np.savez_compressed(ROOT / 'branches.npz', features=x, labels=y, scene_seed=seeds)
    with torch.no_grad(): scores = ranker(torch.from_numpy((x - mean) / std)).numpy()
    metrics = {'reranker': scene_metrics(scores, y, seeds),
               'contact_head': scene_metrics(x[:, :3].mean(1), y, seeds),
               'old_geometric_score': scene_metrics(x[:, 3:6].mean(1), y, seeds),
               'unchanged_command': scene_metrics(-np.tile(np.arange(25), len(y) // 25), y, seeds)}
    paired = {'wins': 0, 'ties': 0, 'losses': 0}
    rows = []
    for seed in np.unique(seeds):
        mask = seeds == seed; labels = y[mask]
        selected = int(scores[mask].argmax()); reranker_hit = bool(labels[selected]); baseline_hit = bool(labels[0])
        if reranker_hit and not baseline_hit: paired['wins'] += 1
        elif reranker_hit == baseline_hit: paired['ties'] += 1
        else: paired['losses'] += 1
        rows.append({'seed': int(seed), 'reranker_choice': selected,
                     'reranker_contact': reranker_hit, 'unchanged_contact': baseline_hit,
                     'positive_candidates': int(labels.sum()), 'average_precision': average_precision(scores[mask], labels)})
    protocol['episode_sha256'] = episode_hashes
    protocol['branch_sha256'] = digest(ROOT / 'branches.npz')
    protocol_path.write_text(json.dumps(protocol, indent=2))
    result = {'protocol_sha256': digest(protocol_path), 'reranker_sha256': digest(V5_ROOT / 'ranker.pt'),
              'scenes': len(rows), 'branches': len(y), 'test_metrics': metrics,
              'paired_reranker_vs_unchanged': paired, 'rows': rows,
              'interpretation': 'Independent offline red-cylinder contact ranking check. No controller use.'}
    (ROOT / 'results.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}, indent=2), flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('action', choices=['run'])
    parser.parse_args(); run()

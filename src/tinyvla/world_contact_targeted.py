"""Run version 4 targeted contact data collection and offline ranking."""
from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pybullet as p
import torch
from torch import nn

from .evaluate import load_policy
from .language import OBJECTS, parse
from .sim import ArmSim
from .world_contact_data import ContactEpisodes, ContactRecordingSim
from .world_contact_rank import average_precision, candidates
from .world_contact_train import ContactHead, calibrate, logits_for, metrics
from .world_data import STATE_OFFSET, STATE_SCALE, VisualFeatures, tip_object
from .world_model import load_world


ROOT = Path('artifacts/world-v4/contact')
INSTRUCTION = 'put the red cylinder on the yellow pad'


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes():
    paths = [Path('configs/world_contact_v4.json'), Path('src/tinyvla/world_contact_targeted.py'),
             Path('src/tinyvla/world_contact_data.py'), Path('src/tinyvla/world_contact_train.py'),
             Path('src/tinyvla/world_contact_rank.py'), Path('src/tinyvla/world_model.py'),
             Path('src/tinyvla/world_data.py'), Path('src/tinyvla/sim.py'),
             Path('src/tinyvla/evaluate.py'), Path('src/tinyvla/language.py'),
             Path('src/tinyvla/controller.py'), Path('configs/world_contact.json'),
             Path('uv.lock'), Path('artifacts/world-v1/model/jepa.pt'), Path('artifacts/model/best.pt'),
             Path('artifacts/world-v3/contact/contact-head.pt')]
    return {str(path): digest(path) for path in paths}


def collect_targeted(config, split: str):
    spec = config['data']
    count = spec[f'{split}_episodes'] if split != 'ranking' else spec['ranking_episodes']
    seed_start = spec['ranking_seed'] if split == 'ranking' else spec[f'{split}_seed']
    folder = ROOT / split
    folder.mkdir(parents=True, exist_ok=True)
    policy, _ = load_policy(config['source']['policy'])
    recorder = ContactRecordingSim(VisualFeatures(policy))
    task = parse(INSTRUCTION)
    rows = []
    started = time.time()
    try:
        for offset in range(count):
            seed = seed_start + offset
            path = folder / f'{seed}.npz'
            meta_path = path.with_suffix('.json')
            if path.exists() and meta_path.exists():
                rows.append(json.loads(meta_path.read_text()))
                continue
            rng = np.random.default_rng(seed)
            recorder.recording = False
            recorder.reset(seed)
            tipped = bool(rng.random() < spec['tip_probability'])
            if tipped:
                tip_object(recorder, task)
            recorder.begin()
            dropped = False
            for attempt in range(2):
                commands = recorder.teacher(task)
                noise = spec['high_noise_std'] if rng.random() < spec['high_noise_probability'] else spec['low_noise_std']
                error = rng.normal(0, noise, 2)
                commands[:4, :2] += error
                commands[1:3, 2] = rng.uniform(.015, .047)
                for phase, command in enumerate(commands):
                    duration = int(rng.choice([120, 150, 190]))
                    if phase in (3, 4, 5):
                        recorder.move_smooth(command[:3], command[3], duration)
                    else:
                        recorder.move(command[:3], command[3], duration)
                    if phase == 3 and attempt == 0 and rng.random() < spec['drop_probability']:
                        recorder.move(recorder.state()[:3], .04, 80)
                        recorder.move(recorder.state()[:3], 0., 60)
                        dropped = True
                if recorder.score(task)['success']:
                    break
                recorder.move(recorder.park, .04, 180)
            recorder.recording = False
            contacts = np.asarray(recorder.contacts, dtype=np.float32)
            contact_binary = contacts > .01
            metadata = {'seed': seed, 'split': split, 'instruction': INSTRUCTION,
                        'object': task.object, 'transitions': len(recorder.actions),
                        'tipped': tipped, 'drop_commands': dropped,
                        'success': recorder.score(task)['success'],
                        'contact_steps': contact_binary.sum(axis=0).tolist(),
                        'positive_steps': int(contact_binary.sum()), 'force_threshold_n': .01,
                        'high_noise_std': noise}
            np.savez_compressed(path, features=np.asarray(recorder.features, np.float16),
                                actions=np.asarray(recorder.actions, np.float32),
                                times=np.asarray(recorder.times, np.float64),
                                contact_force_n=contacts, seed=seed)
            meta_path.write_text(json.dumps(metadata, indent=2))
            rows.append(metadata)
            if (offset + 1) % 20 == 0:
                print(json.dumps({'split': split, 'episodes': offset + 1,
                                  'seconds': round(time.time() - started, 1)}), flush=True)
    finally:
        recorder.close()
    summary = {'split': split, 'episodes': len(rows), 'seed_min': seed_start,
               'seed_max': seed_start + count - 1,
               'transitions': sum(row['transitions'] for row in rows),
               'episodes_with_contact': sum(row['positive_steps'] > 0 for row in rows),
               'tipped_episodes': sum(row['tipped'] for row in rows),
               'drop_episodes': sum(row['drop_commands'] for row in rows),
               'contact_steps_by_object': np.sum([row['contact_steps'] for row in rows], axis=0).tolist(),
               'rows': rows}
    (ROOT / f'{split}_data.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps({key: value for key, value in summary.items() if key != 'rows'}), flush=True)
    return summary


def merge_episodes(base_folder: Path, targeted_folder: Path, horizon: int):
    base = ContactEpisodes(base_folder, horizon)
    extra = ContactEpisodes(targeted_folder, horizon)
    merged = ContactEpisodes.__new__(ContactEpisodes)
    merged.features = torch.cat((base.features, extra.features))
    merged.actions = torch.cat((base.actions, extra.actions))
    merged.contacts = torch.cat((base.contacts, extra.contacts))
    merged.starts = torch.cat((base.starts, extra.starts + len(base.features)))
    merged.seeds = base.seeds + extra.seeds
    merged.horizon = horizon
    if set(base.seeds) & set(extra.seeds):
        raise ValueError('Base and targeted scene seeds overlap.')
    return merged


def train(config):
    torch.manual_seed(config['training']['seed'])
    np.random.seed(config['training']['seed'])
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(config['training']['threads'])
    base_data = Path(config['source']['base_contact_data'])
    train_data = merge_episodes(base_data / 'train', ROOT / 'train', config['data']['horizon'])
    validation = merge_episodes(base_data / 'validation', ROOT / 'validation', config['data']['horizon'])
    testing = merge_episodes(base_data / 'test', ROOT / 'test', config['data']['horizon'])
    base, base_checkpoint = load_world(config['source']['checkpoint'])
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    head = ContactHead(base_checkpoint['config']['training']['latent_size'], config['training']['hidden_size'])
    target_ids = torch.cat([train_data.starts[:, None] + i for i in (1, 2, 3)]).reshape(-1)
    prevalence = train_data.contacts[target_ids].mean(0).clamp_min(.001)
    pos_weight = ((1 - prevalence) / prevalence).clamp(1, 50)
    optimizer = torch.optim.AdamW(head.parameters(), lr=config['training']['learning_rate'])
    log_path = ROOT / 'training.jsonl'
    log_path.write_text('')
    best_ap, best_state = -1., None
    started = time.time()
    for step in range(1, config['training']['steps'] + 1):
        head.train()
        ids = torch.randint(len(train_data.starts), (config['training']['batch_size'],))
        history, _, actions, labels = train_data.batch(ids)
        with torch.no_grad():
            states, _ = base.rollout(base.encode(history), actions)
        logits = head(states.permute(1, 0, 2, 3))
        target = labels[:, None].expand(-1, 3, -1, -1)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
        optimizer.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(head.parameters(), 5); optimizer.step()
        if step in (1, config['training']['steps']) or step % 100 == 0:
            head.eval()
            with torch.no_grad():
                val_logits, val_labels, _ = logits_for(base, head, validation, examples=3000)
                score = metrics(val_logits.sigmoid().mean(0), val_labels)
            record = {'step': step, 'loss': float(loss.detach()), 'validation_uncalibrated': score,
                      'seconds': round(time.time() - started, 1)}
            with log_path.open('a') as stream:
                stream.write(json.dumps(record) + '\n')
            if score['macro_average_precision'] > best_ap:
                best_ap = score['macro_average_precision']
                best_state = {key: value.detach().clone() for key, value in head.state_dict().items()}
            print(json.dumps({'step': step, 'validation_ap': score['macro_average_precision'],
                              'validation_brier': score['macro_brier']}), flush=True)
    head.load_state_dict(best_state); head.eval()
    val_logits, val_labels, val_old = logits_for(base, head, validation)
    scale, bias = calibrate(val_logits, val_labels, config['training']['calibration_steps'])
    test_logits, test_labels, test_old = logits_for(base, head, testing)
    calibrated = metrics((test_logits.mean(0) * scale + bias).sigmoid(), test_labels)
    targeted_test = ContactEpisodes(ROOT / 'test', config['data']['horizon'])
    targeted_logits, targeted_labels, targeted_old = logits_for(base, head, targeted_test)
    targeted_metrics = metrics((targeted_logits.mean(0) * scale + bias).sigmoid(), targeted_labels)
    v3_path = Path(config['source']['base_contact_data']) / 'contact-head.pt'
    v3_bundle = torch.load(v3_path, map_location='cpu', weights_only=False)
    v3_head = ContactHead(v3_bundle['latent_size'], v3_bundle['hidden_size'])
    v3_head.load_state_dict(v3_bundle['state_dict']); v3_head.eval()
    v3_logits, _, _ = logits_for(base, v3_head, targeted_test)
    v3_scores = (v3_logits.mean(0) * torch.tensor(v3_bundle['calibration_scale']) +
                 torch.tensor(v3_bundle['calibration_bias'])).sigmoid()
    dataset_hashes = {}
    for label, folder in [('base_train', base_data / 'train'), ('base_validation', base_data / 'validation'),
                          ('base_test', base_data / 'test'), ('target_train', ROOT / 'train'),
                          ('target_validation', ROOT / 'validation'), ('target_test', ROOT / 'test')]:
        dataset_hashes[label] = {path.name: digest(path) for path in sorted(folder.glob('*.npz'))}
    report = {'version': 4, 'source_hashes': source_hashes(),
              'data': {'train': {'episodes': len(train_data.seeds), 'windows': len(train_data.starts), 'seeds': train_data.seeds},
                       'validation': {'episodes': len(validation.seeds), 'windows': len(validation.starts), 'seeds': validation.seeds},
                       'test': {'episodes': len(testing.seeds), 'windows': len(testing.starts), 'seeds': testing.seeds}},
              'selected_step': max((json.loads(line) for line in log_path.read_text().splitlines()),
                                   key=lambda row: row['validation_uncalibrated']['macro_average_precision'])['step'],
              'calibration': {'scale': scale.tolist(), 'bias': bias.tolist()},
              'validation_calibrated': metrics((val_logits.mean(0) * scale + bias).sigmoid(), val_labels),
              'test_calibrated': calibrated, 'old_geometric_held_score_test': metrics(test_old, test_labels),
              'targeted_red_cylinder_test': targeted_metrics,
              'version_3_head_on_targeted_red_cylinder_test': metrics(v3_scores, targeted_labels),
              'targeted_red_cylinder_old_score_test': metrics(targeted_old, targeted_labels),
              'dataset_sha256': dataset_hashes,
              'test_uncalibrated': metrics(test_logits.sigmoid().mean(0), test_labels),
              'parameters': sum(parameter.numel() for parameter in head.parameters()),
              'seconds': time.time() - started,
              'interpretation': 'Targeted offline contact prediction only. The contact head did not control the arm.'}
    checkpoint_path = ROOT / 'contact-head.pt'
    torch.save({'state_dict': head.state_dict(), 'latent_size': base_checkpoint['config']['training']['latent_size'],
                'hidden_size': config['training']['hidden_size'], 'calibration_scale': scale.tolist(),
                'calibration_bias': bias.tolist(), 'training_seeds': train_data.seeds,
                'validation_seeds': validation.seeds, 'contact_threshold_n': .01,
                'contact_channels': ['red cube', 'blue cube', 'green cylinder', 'red cylinder']}, checkpoint_path)
    report['checkpoint_sha256'] = digest(checkpoint_path)
    (ROOT / 'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'selected_step': report['selected_step'], 'test': calibrated,
                      'old_score': report['old_geometric_held_score_test']}, indent=2), flush=True)
    return report


def rank(config):
    if (ROOT / 'ranking_results.json').exists():
        raise FileExistsError('The version 4 ranking result already exists. Preserve it and use fresh seeds for another audit.')
    head_path = ROOT / 'contact-head.pt'
    model_path = Path(config['source']['checkpoint'])
    base, _ = load_world(model_path)
    bundle = torch.load(head_path, map_location='cpu', weights_only=False)
    head = ContactHead(bundle['latent_size'], bundle['hidden_size'])
    head.load_state_dict(bundle['state_dict']); head.eval()
    scale, bias = torch.tensor(bundle['calibration_scale']), torch.tensor(bundle['calibration_bias'])
    protocol = {'seed_start': config['data']['ranking_seed'],
                'seed_end': config['data']['ranking_seed'] + config['data']['ranking_episodes'] - 1,
                'scenes': config['data']['ranking_episodes'], 'candidates_per_scene': 25,
                'object': 'red cylinder', 'tip_probability': config['data']['tip_probability'],
                'contact_threshold_n': .01, 'branches': 'offline state copies; no controller use',
                'source_hashes': {**source_hashes(), str(head_path): digest(head_path)},
                'contact_report_sha256': digest(ROOT / 'report.json')}
    (ROOT / 'ranking_protocol.json').write_text(json.dumps(protocol, indent=2))
    collect_targeted(config, 'ranking')
    rows = []
    for path in sorted((ROOT / 'ranking').glob('*.npz')):
        with np.load(path) as record:
            features, actions, seed = record['features'].astype(np.float32), record['actions'].astype(np.float32), int(record['seed'])
        metadata = json.loads(path.with_suffix('.json').read_text())
        task = parse(metadata['instruction']); source = OBJECTS.index(task.object)
        choice = next((index for index in range(1, len(actions) - 3)
                       if actions[index, 2] <= .055 and actions[index, 3] > .02), None)
        if choice is None:
            raise ValueError(f'No open-gripper candidate point found in seed {seed}.')
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
            predicted = (logits.mean(0)[:, 1, source] * scale[source] + bias[source]).sigmoid().numpy()
            old = outcomes[:, :, 1, source * 8 + 7].mean(0).numpy().clip(0, 1)
        sim = ArmSim(); actual = []
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
            for candidate in sequence:
                p.restoreState(state_id); sim.ticks, sim.command_count = ticks, count; sim.max_heights = max_heights.copy()
                for action_index, action in enumerate(candidate):
                    sim.move(action[:3], action[3], round(float(action[4]) * 240))
                    if action_index == 1:
                        break
                force = sum(max(0., float(contact[9])) for contact in p.getContactPoints(
                    bodyA=sim.arm, bodyB=sim.objects[task.object]) if contact[3] in (9, 10))
                actual.append(force > .01)
            p.removeState(state_id)
        finally:
            sim.close()
        actual = np.asarray(actual, bool)
        head_pick, old_pick = int(predicted.argmax()), int(old.argmax())
        rows.append({'seed': seed, 'object': task.object, 'tipped': metadata['tipped'],
                     'replay_error': replay_error, 'contact_count': int(actual.sum()),
                     'head_top1_contact': bool(actual[head_pick]), 'unchanged_contact': bool(actual[0]),
                     'old_top1_contact': bool(actual[old_pick]), 'oracle_contact': bool(actual.any()),
                     'head_ap': average_precision(predicted, actual), 'old_ap': average_precision(old, actual),
                     'actual_contact': actual.tolist(), 'predicted_contact': predicted.tolist()})
        print(json.dumps({'seed': seed, 'tipped': metadata['tipped'], 'contacts': int(actual.sum()),
                          'head_choice_contact': bool(actual[head_pick]), 'unchanged_contact': bool(actual[0])}), flush=True)
    if len(rows) != config['data']['ranking_episodes']:
        raise ValueError(f'Expected {config["data"]["ranking_episodes"]} ranking rows, found {len(rows)}.')
    output = {'protocol': protocol, 'scenes': len(rows), 'branches': len(rows) * 25,
              'head_top1_rate': float(np.mean([row['head_top1_contact'] for row in rows])),
              'unchanged_rate': float(np.mean([row['unchanged_contact'] for row in rows])),
              'old_top1_rate': float(np.mean([row['old_top1_contact'] for row in rows])),
              'oracle_rate': float(np.mean([row['oracle_contact'] for row in rows])),
              'head_mean_ap': float(np.nanmean([row['head_ap'] for row in rows])),
              'old_mean_ap': float(np.nanmean([row['old_ap'] for row in rows])), 'rows': rows,
              'interpretation': 'Fresh red-cylinder-only simulation ranking audit. No controller integration.'}
    (ROOT / 'ranking_results.json').write_text(json.dumps(output, indent=2))
    return output


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['collect', 'train', 'rank', 'all'])
    parser.add_argument('--split', choices=['train', 'validation', 'test'])
    args = parser.parse_args()
    config = json.loads(Path('configs/world_contact_v4.json').read_text())
    ROOT.mkdir(parents=True, exist_ok=True)
    if args.action == 'collect':
        if args.split not in ('train', 'validation', 'test'):
            parser.error('collect needs --split train, validation, or test')
        collect_targeted(config, args.split)
    elif args.action == 'train':
        train(config)
    elif args.action == 'rank':
        print(json.dumps(rank(config), indent=2))
    else:
        for split in ('train', 'validation', 'test'):
            collect_targeted(config, split)
        train(config)
        print(json.dumps(rank(config), indent=2))


if __name__ == '__main__':
    main()

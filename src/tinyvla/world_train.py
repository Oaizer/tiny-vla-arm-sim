"""Train future prediction and measure it on separate simulation episodes."""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .world_data import XYZ_SCALE, STATE_SCALE
from .world_model import WorldModel, load_world


class Episodes:
    def __init__(self, folder, horizon=3):
        features, targets, actions, starts = [], [], [], []
        self.seeds = []
        offset = 0
        for path in sorted(Path(folder).glob('*.npz')):
            with np.load(path) as data:
                feature = data['features'].astype(np.float32)
                target = data['targets'].astype(np.float32)
                objects = target[:, :32].reshape(-1, 4, 8)
                xyz = objects[..., :3] * XYZ_SCALE + np.array([.5, 0, .02])
                inside = ((xyz >= [.25, -.4, 0]) & (xyz <= [.8, .4, .4])).all(-1)
                objects[..., 6] = inside
                action = data['actions'].astype(np.float32)
                if not np.allclose(np.diff(data['times']), action[:, 4], atol=1e-6):
                    raise ValueError(f'Command times do not match the recorded transition in {path}.')
                self.seeds.append(int(data['seed']))
            if len(feature) != len(action) + 1:
                raise ValueError(f'An observation is missing from {path}.')
            features.append(feature)
            targets.append(target)
            actions.append(np.concatenate([action, np.zeros((1, 5), np.float32)]))
            starts.extend(range(offset + 2, offset + len(feature) - horizon))
            offset += len(feature)
        if not features:
            raise ValueError(f'No episodes were found in {folder}.')
        self.features = torch.from_numpy(np.concatenate(features))
        self.targets = torch.from_numpy(np.concatenate(targets))
        self.actions = torch.from_numpy(np.concatenate(actions))
        self.starts = torch.tensor(starts, dtype=torch.long)
        self.horizon = horizon

    def batch(self, ids):
        now = self.starts[ids]
        history = now[:, None] + torch.arange(-2, 1)
        future = now[:, None] + torch.arange(1, self.horizon + 1)
        future_history = future[:, :, None] + torch.arange(-2, 1)
        actions = self.actions[now[:, None] + torch.arange(self.horizon)]
        return self.features[history], self.features[future_history], actions, self.targets[now], self.targets[future]


def outcome_loss(predicted, expected):
    weights = predicted.new_tensor(([2, 2, 4, .1, .1, .1, .5, 2] * 4) + [1, 1, 1, 1])
    expected = expected.expand_as(predicted)
    mask = torch.ones_like(expected)
    for obj in range(4):
        mask[..., obj * 8:obj * 8 + 6] = expected[..., obj * 8 + 6:obj * 8 + 7]
    return (F.smooth_l1_loss(predicted, expected, reduction='none') * weights * mask).mean()


@torch.inference_mode()
def measure(model, data, shuffled=False, max_examples=None):
    total = min(len(data.starts), max_examples or len(data.starts))
    ids = torch.randperm(len(data.starts), generator=torch.Generator().manual_seed(99))[:total]
    xyz, state_errors, latent_errors, held_errors, persistent = [], [], [], [], []
    visual_persistent, latent_persistent, moving_errors, moving_counts = [], [], [], []
    all_xyz, presence_errors, latent_values = [], [], []
    for chunk in ids.split(256):
        history, future, action, now_label, expected = data.batch(chunk)
        if shuffled:
            action = action.roll(1, 0)
        latent = model.encode(history)
        latent_values.append(latent[:, 25:])
        current_outcome = model.readout(latent)
        predicted_latent, predicted = model.rollout(latent, action)
        target_latent = model.encode(future.flatten(0, 1), target=True).reshape(len(chunk), data.horizon, -1)
        mean = predicted.mean(0)
        po = mean[..., :32].reshape(len(chunk), data.horizon, 4, 8)
        truth = expected[..., :32].reshape(len(chunk), data.horizon, 4, 8)
        error = ((po[..., :3] - truth[..., :3]) * torch.tensor(XYZ_SCALE)).square().mean(3)
        valid = truth[..., 6]
        xyz.append((error * valid).sum(2) / valid.sum(2).clamp_min(1))
        all_xyz.append(error.mean(2))
        presence_errors.append((po[..., 6] - valid).square().mean(2))
        held_errors.append((po[..., 7] - truth[..., 7]).square().mean(2))
        state_errors.append(((mean[..., 32:] - expected[..., 32:]) * torch.tensor(STATE_SCALE)).square().mean(2))
        latent_errors.append((predicted_latent.mean(0) - target_latent).square().mean(2))
        original = now_label[:, None, :32].reshape(len(chunk), 1, 4, 8)
        no_change_errors = ((original[..., :3] - truth[..., :3]) * torch.tensor(XYZ_SCALE)).square().mean(3)
        persistent.append((no_change_errors * valid).sum(2) / valid.sum(2).clamp_min(1))
        observed = current_outcome[:, None, :32].reshape(len(chunk), 1, 4, 8)
        visual_error = ((observed[..., :3] - truth[..., :3]) * torch.tensor(XYZ_SCALE)).square().mean(3)
        visual_persistent.append((visual_error * valid).sum(2) / valid.sum(2).clamp_min(1))
        current_target = model.encode(history, target=True)
        latent_persistent.append((current_target[:, None] - target_latent).square().mean(2))
        moving = (no_change_errors.sqrt() > .01) & (valid > .5)
        moving_errors.append((((po[..., :3] - truth[..., :3]) * torch.tensor(XYZ_SCALE)).square().mean(3) * moving).sum((0, 2)))
        moving_counts.append(moving.sum((0, 2)))
    result = {'examples': total, 'episodes': len(data.seeds), 'shuffled_actions': shuffled}
    for label, values, scale, square_root in [('object_xyz_rmse_mm', xyz, 1000, True),
                                             ('sensor_rmse_m', state_errors, 1, True),
                                             ('latent_mse', latent_errors, 1, False),
                                             ('held_score_mse', held_errors, 1, False),
                                             ('oracle_no_change_xyz_rmse_mm', persistent, 1000, True),
                                             ('visual_no_change_xyz_rmse_mm', visual_persistent, 1000, True),
                                             ('latent_no_change_mse', latent_persistent, 1, False)]:
        means = torch.cat(values).mean(0)
        result[label] = ((means.sqrt() if square_root else means) * scale).tolist()
    counts = torch.stack(moving_counts).sum(0)
    result['moving_object_xyz_rmse_mm'] = (torch.stack(moving_errors).sum(0).div(counts.clamp_min(1)).sqrt() * 1000).tolist()
    result['moving_object_count'] = counts.tolist()
    result['unbounded_all_object_xyz_rmse_mm'] = (torch.cat(all_xyz).mean(0).sqrt() * 1000).tolist()
    result['workspace_presence_mse'] = torch.cat(presence_errors).mean(0).tolist()
    result['learned_latent_mean_std'] = float(torch.cat(latent_values).std(0).mean())
    return result


def train(config, root):
    root = Path(root)
    output = root / 'model'
    output.mkdir(exist_ok=True)
    settings = config['training']
    torch.manual_seed(settings['seed'])
    np.random.seed(settings['seed'])
    torch.use_deterministic_algorithms(True)
    training = Episodes(root / 'trajectories/train')
    validation = Episodes(root / 'trajectories/validation')
    if set(training.seeds) & set(validation.seeds):
        raise ValueError('Training and validation share episode seeds.')
    model = WorldModel(config)
    model.mean.copy_(training.features.mean(0))
    model.std.copy_(training.features.std(0).clamp_min(.05))
    begin = time.time()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=settings['learning_rate'])
    # Fit useful visual features before freezing them for the first dynamics model.
    for step in range(400):
        ids = torch.randint(len(training.starts), (settings['batch_size'],))
        history, _, _, target, _ = training.batch(ids)
        loss = outcome_loss(model.readout(model.encode(history)), target)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    model.target_encoder.load_state_dict(model.encoder.state_dict())
    print(json.dumps({'warmup_steps': 400, 'seconds': round(time.time() - begin, 1)}), flush=True)
    rows = []
    (output / 'training.jsonl').write_text('')
    for stage, steps in [('fixed', settings['fixed_steps']), ('jepa', settings['jepa_steps'])]:
        if stage == 'jepa':
            model, _ = load_world(output / 'fixed.pt')
        model.encoder.requires_grad_(stage == 'jepa')
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=settings['learning_rate'])
        best = float('inf')
        for step in range(1, steps + 1):
            model.train()
            ids = torch.randint(len(training.starts), (settings['batch_size'],))
            history, future, action, current_label, next_label = training.batch(ids)
            latent = model.encode(history)
            predictions, outcomes = model.rollout(latent, action)
            with torch.no_grad():
                target = model.encode(future.flatten(0, 1), target=True).reshape(len(ids), training.horizon, -1)
            prediction_loss = (predictions - target[None]).square().mean() + 4 * (predictions[..., :25] - target[None, ..., :25]).square().mean()
            semantic_loss = outcome_loss(outcomes, next_label[None]) + outcome_loss(model.readout(latent), current_label)
            variance_loss = F.relu(.5 - latent[:, 25:].std(0)).mean()
            loss = prediction_loss + 12 * semantic_loss + .2 * variance_loss
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            if stage == 'jepa':
                model.update_target(settings['ema'])
            if step == 1 or step % 200 == 0 or step == steps:
                model.eval()
                metrics = measure(model, validation, max_examples=1500)
                # Use a fixed physical-unit measure to select models as the target encoder changes.
                score = float(np.mean(metrics['object_xyz_rmse_mm']))
                row = {'stage': stage, 'step': step, 'loss': float(loss.detach()), 'validation': metrics,
                       'seconds': round(time.time() - begin, 1)}
                rows.append(row)
                with (output / 'training.jsonl').open('a') as log:
                    log.write(json.dumps(row) + '\n')
                print(json.dumps({'stage': stage, 'step': step, 'validation_xyz_mm': round(score, 2),
                                  'seconds': row['seconds']}), flush=True)
                if score < best:
                    best = score
                    torch.save({'state_dict': model.state_dict(), 'config': config, 'stage': stage,
                                'step': step, 'validation': metrics, 'training_seconds': time.time() - begin,
                                'training_seeds': training.seeds, 'validation_seeds': validation.seeds}, output / f'{stage}.pt')
    (output / 'training_summary.json').write_text(json.dumps({'seconds': time.time() - begin, 'records': rows}, indent=2))


def evaluate_prediction(config, root):
    root = Path(root)
    testing = Episodes(root / 'trajectories/test')
    result = {}
    for stage in ['fixed', 'jepa']:
        path = root / 'model' / f'{stage}.pt'
        model, checkpoint = load_world(path)
        if set(testing.seeds) & set(checkpoint['training_seeds'] + checkpoint['validation_seeds']):
            raise ValueError('Prediction tests share training or validation scenes.')
        result[stage] = {'normal': measure(model, testing), 'shuffled_actions': measure(model, testing, shuffled=True),
                         'checkpoint_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                         'parameters_without_target_encoder': sum(p.numel() for name, p in model.named_parameters()
                                                                 if not name.startswith('target_encoder.')),
                         'bytes': path.stat().st_size}
        print(json.dumps({stage: result[stage]}), flush=True)
    (root / 'prediction_results.json').write_text(json.dumps(result, indent=2))
    return result

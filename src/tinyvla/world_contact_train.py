"""Train and calibrate a contact head on held-out PyBullet scenes."""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .world_contact_data import ContactEpisodes, source_hashes
from .world_model import load_world

OBJECTS = ['red cube', 'blue cube', 'green cylinder', 'red cylinder']


class ContactHead(nn.Module):
    def __init__(self, latent, hidden):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(latent, hidden), nn.GELU(), nn.Linear(hidden, 4))

    def forward(self, value):
        return self.layers(value)


@torch.inference_mode()
def logits_for(base, head, data, examples=None):
    ids = torch.arange(len(data.starts))
    if examples and len(ids) > examples:
        ids = ids[::max(1, len(ids) // examples)][:examples]
    logits, labels, geometric_held = [], [], []
    for batch_ids in ids.split(256):
        history, _, actions, contact = data.batch(batch_ids)
        states, outcomes = base.rollout(base.encode(history), actions)
        # All three learned predictors vote. Keep their predictions available for
        # the disagreement measure; the mean score is calibrated on validation.
        logits.append(head(states.permute(1, 0, 2, 3)).permute(1, 0, 2, 3).cpu())
        labels.append(contact.cpu())
        held = torch.stack([outcomes[:, :, :, index * 8 + 7] for index in range(4)], -1)
        geometric_held.append(held.mean(0).cpu())
    return torch.cat(logits, 1), torch.cat(labels, 0), torch.cat(geometric_held, 0)


def calibrate(raw, labels, steps=300):
    # Fit one positive scale and one bias per object on validation scenes only.
    # Validation batches are produced under inference_mode; clone them after
    # leaving that context so LBFGS can use them in its backward pass.
    raw = raw.detach().float().mean(0).reshape(-1, 4).clone()
    labels = labels.float().reshape(-1, 4).clone()
    slope = nn.Parameter(torch.ones(4))
    bias = nn.Parameter(torch.zeros(4))
    optimizer = torch.optim.LBFGS([slope, bias], lr=.2, max_iter=steps,
                                  line_search_fn='strong_wolfe')
    def closure():
        optimizer.zero_grad()
        scale = slope.clamp(.05, 10)
        loss = nn.functional.binary_cross_entropy_with_logits(raw * scale + bias, labels)
        loss.backward()
        return loss
    optimizer.step(closure)
    return slope.detach().clamp(.05, 10), bias.detach()


def metrics(scores, labels):
    scores = scores.reshape(-1, 4).clamp(1e-6, 1 - 1e-6)
    labels = labels.reshape(-1, 4)
    result = {'examples': len(labels), 'objects': {}}
    for column, name in enumerate(OBJECTS):
        y, p = labels[:, column], scores[:, column]
        positives = int(y.sum())
        prediction = p >= .5
        tp = int((prediction & (y > .5)).sum())
        fp = int((prediction & (y <= .5)).sum())
        fn = int((~prediction & (y > .5)).sum())
        order = torch.argsort(p, descending=True)
        ranked = y[order]
        precision = ranked.cumsum(0) / torch.arange(1, len(ranked) + 1)
        ap = float((precision * ranked).sum() / max(1, positives))
        edges = torch.linspace(0, 1, 11)
        ece = 0.0
        for low, high in zip(edges[:-1], edges[1:]):
            mask = (p >= low) & (p < high if high < 1 else p <= high)
            if mask.any():
                ece += float(mask.float().mean() * abs(p[mask].mean() - y[mask].mean()))
        result['objects'][name] = {
            'positive_rate': float(y.mean()), 'average_precision': ap,
            'brier': float((p - y).square().mean()), 'no_contact_brier': float(y.mean()),
            'expected_calibration_error_10_bins': ece,
            'threshold': .5, 'precision': tp / max(1, tp + fp),
            'recall': tp / max(1, tp + fn), 'true_positive': tp,
            'false_positive': fp, 'false_negative': fn}
    result['macro_average_precision'] = float(np.mean([v['average_precision'] for v in result['objects'].values()]))
    result['macro_brier'] = float(np.mean([v['brier'] for v in result['objects'].values()]))
    result['macro_no_contact_brier'] = float(np.mean([v['no_contact_brier'] for v in result['objects'].values()]))
    return result


def train(config, root):
    torch.manual_seed(config['training']['seed'])
    np.random.seed(config['training']['seed'])
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(config['training']['threads'])
    root = Path(root)
    training = ContactEpisodes(root / 'train', config['data']['horizon'])
    validation = ContactEpisodes(root / 'validation', config['data']['horizon'])
    testing = ContactEpisodes(root / 'test', config['data']['horizon'])
    if set(training.seeds) & set(validation.seeds) or set(training.seeds + validation.seeds) & set(testing.seeds):
        raise ValueError('Contact training, validation, and test episode seeds overlap.')
    base, base_checkpoint = load_world(config['source']['checkpoint'])
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    head = ContactHead(base_checkpoint['config']['training']['latent_size'],
                       config['training']['hidden_size'])
    positive = training.contacts[torch.cat([training.starts[:, None] + 1,
                                             training.starts[:, None] + 2,
                                             training.starts[:, None] + 3]).reshape(-1)]
    prevalence = positive.mean(0).clamp_min(.001)
    pos_weight = ((1 - prevalence) / prevalence).clamp(1, 50)
    optimizer = torch.optim.AdamW(head.parameters(), lr=config['training']['learning_rate'])
    batch_size, steps = config['training']['batch_size'], config['training']['steps']
    begin = time.time()
    log_path = root / 'training.jsonl'
    log_path.write_text('')
    best_ap, best_state = -1, None
    for step in range(1, steps + 1):
        head.train()
        ids = torch.randint(len(training.starts), (batch_size,))
        history, _, actions, labels = training.batch(ids)
        with torch.no_grad():
            states, _ = base.rollout(base.encode(history), actions)
        predictions = head(states.permute(1, 0, 2, 3))
        target = labels[:, None].expand(-1, 3, -1, -1)
        loss = nn.functional.binary_cross_entropy_with_logits(predictions, target,
                                                               pos_weight=pos_weight)
        optimizer.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(head.parameters(), 5); optimizer.step()
        if step in (1, steps) or step % 100 == 0:
            head.eval()
            with torch.no_grad():
                val_logits, val_labels, _ = logits_for(base, head, validation, examples=3000)
                val_metrics = metrics(val_logits.sigmoid().mean(0), val_labels)
            record = {'step': step, 'loss': float(loss.detach()),
                      'validation_uncalibrated': val_metrics, 'seconds': round(time.time() - begin, 1)}
            with log_path.open('a') as log:
                log.write(json.dumps(record) + '\n')
            print(json.dumps({'step': step, 'validation_ap': val_metrics['macro_average_precision'],
                              'validation_brier': val_metrics['macro_brier']}), flush=True)
            if val_metrics['macro_average_precision'] > best_ap:
                best_ap = val_metrics['macro_average_precision']
                best_state = {key: value.detach().clone() for key, value in head.state_dict().items()}
    head.load_state_dict(best_state); head.eval()
    val_logits, val_labels, val_geometric = logits_for(base, head, validation)
    scale, bias = calibrate(val_logits, val_labels, config['training']['calibration_steps'])
    test_logits, test_labels, test_geometric = logits_for(base, head, testing)
    uncalibrated = metrics(test_logits.sigmoid().mean(0), test_labels)
    calibrated_scores = (test_logits.mean(0) * scale + bias).sigmoid()
    calibrated = metrics(calibrated_scores, test_labels)
    val_calibrated = metrics((val_logits.mean(0) * scale + bias).sigmoid(), val_labels)
    baseline_test = metrics(test_geometric, test_labels)
    baseline_validation = metrics(val_geometric, val_labels)
    ckpt_path = root / 'contact-head.pt'
    payload = {'state_dict': head.state_dict(), 'latent_size': base_checkpoint['config']['training']['latent_size'],
               'hidden_size': config['training']['hidden_size'], 'calibration_scale': scale.tolist(),
               'calibration_bias': bias.tolist(), 'source_checkpoint_sha256': hashlib.sha256(Path(config['source']['checkpoint']).read_bytes()).hexdigest(),
               'training_seeds': training.seeds, 'validation_seeds': validation.seeds,
               'contact_threshold_n': .01, 'contact_channels': OBJECTS}
    torch.save(payload, ckpt_path)
    report = {'source_hashes': source_hashes(config), 'base_checkpoint_sha256': payload['source_checkpoint_sha256'],
              'data': {name: {'episodes': len(ds.seeds), 'seeds': ds.seeds,
                              'positive_rate': ds.contacts.mean(0).tolist(), 'windows': len(ds.starts)}
                       for name, ds in [('train', training), ('validation', validation), ('test', testing)]},
              'selected_step': best_step(log_path), 'calibration': {'scale': scale.tolist(), 'bias': bias.tolist()},
              'validation_calibrated': val_calibrated, 'test_uncalibrated': uncalibrated,
              'test_calibrated': calibrated, 'old_geometric_held_score_validation': baseline_validation,
              'old_geometric_held_score_test': baseline_test,
              'checkpoint_sha256': hashlib.sha256(ckpt_path.read_bytes()).hexdigest(),
              'parameters': sum(p.numel() for p in head.parameters()), 'seconds': time.time() - begin,
              'interpretation': 'Offline prediction test only. The contact model did not control the arm.'}
    (root / 'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'report': report}, indent=2), flush=True)
    return report


def best_step(path):
    records = [json.loads(line) for line in Path(path).read_text().splitlines()]
    return max(records, key=lambda row: row['validation_uncalibrated']['macro_average_precision'])['step']

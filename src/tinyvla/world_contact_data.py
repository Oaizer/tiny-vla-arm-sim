"""Record and load direct PyBullet finger-contact labels for world-model training."""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from .evaluate import instruction_for, load_policy
from .language import parse
from .sim import ArmSim
from .world_data import RecordingSim, VisualFeatures, tip_object


class ContactRecordingSim(RecordingSim):
    """Save fingertip force labels after each actual servo command."""

    def begin(self):
        self.contacts = []
        super().begin()

    def capture(self):
        observation = self.observation()
        self.features.append(self.extractor(observation))
        values = []
        for name in ('red cube', 'blue cube', 'green cylinder', 'red cylinder'):
            force = 0.0
            if name in self.objects:
                for contact in self._finger_contacts(self.objects[name]):
                    force += max(0.0, float(contact[9]))
            values.append(force)
        self.contacts.append(values)
        self.times.append(self.ticks / 240)

    def _finger_contacts(self, body):
        import pybullet as p
        return (contact for contact in p.getContactPoints(bodyA=self.arm, bodyB=body)
                if contact[3] in (9, 10))


def collect(config, output, policy_path, split, count=None):
    output = Path(output)
    folder = output / split
    folder.mkdir(parents=True, exist_ok=True)
    spec = config['data']
    count = count or spec[f'{split}_episodes']
    offset = spec[f'{split}_offset']
    start = spec[f'{split}_seed']
    policy, _ = load_policy(policy_path)
    recorder = ContactRecordingSim(VisualFeatures(policy))
    summaries = []
    begin = time.time()
    try:
        for local_index in range(count):
            index = offset + local_index
            seed = start + index
            path = folder / f'{seed}.npz'
            meta_path = path.with_suffix('.json')
            if path.exists() and meta_path.exists():
                summaries.append(json.loads(meta_path.read_text()))
                continue
            rng = np.random.default_rng(seed)
            task = parse(instruction_for(index))
            recorder.recording = False
            recorder.reset(seed)
            tipped = bool('cylinder' in task.object and rng.random() < .65)
            if tipped:
                tip_object(recorder, task)
            recorder.begin()
            dropped = False
            for attempt in range(2):
                commands = recorder.teacher(task)
                error = rng.normal(0, .016 if rng.random() < .45 else .004, 2)
                commands[:4, :2] += error
                commands[1:3, 2] = rng.uniform(.015, .047)
                for phase, command in enumerate(commands):
                    duration = int(rng.choice([120, 150, 190]))
                    if phase in (3, 4, 5):
                        recorder.move_smooth(command[:3], command[3], duration)
                    else:
                        recorder.move(command[:3], command[3], duration)
                    if phase == 3 and attempt == 0 and rng.random() < .30:
                        recorder.move(recorder.state()[:3], .04, 80)
                        recorder.move(recorder.state()[:3], 0., 60)
                        dropped = True
                if recorder.score(task)['success']:
                    break
                recorder.move(recorder.park, .04, 180)
            recorder.recording = False
            contacts = np.asarray(recorder.contacts, dtype=np.float32)
            contact_binary = contacts > .01
            metadata = {'seed': seed, 'index': index, 'split': split,
                        'instruction': task.instruction, 'object': task.object,
                        'transitions': len(recorder.actions), 'tipped': tipped,
                        'drop_commands': dropped, 'success': recorder.score(task)['success'],
                        'contact_steps': contact_binary.sum(axis=0).tolist(),
                        'positive_steps': int(contact_binary.sum()),
                        'force_threshold_n': .01}
            np.savez_compressed(path, features=np.asarray(recorder.features, np.float16),
                                actions=np.asarray(recorder.actions, np.float32),
                                times=np.asarray(recorder.times, np.float64),
                                contact_force_n=contacts, seed=seed)
            meta_path.write_text(json.dumps(metadata, indent=2))
            summaries.append(metadata)
            if (local_index + 1) % 20 == 0:
                print(json.dumps({'split': split, 'episodes': local_index + 1,
                                  'seconds': round(time.time() - begin, 1)}), flush=True)
    finally:
        recorder.close()
    summary = {'split': split, 'episodes': len(summaries), 'transitions': sum(x['transitions'] for x in summaries),
               'seed_min': min(x['seed'] for x in summaries), 'seed_max': max(x['seed'] for x in summaries),
               'episodes_with_contact': sum(x['positive_steps'] > 0 for x in summaries),
               'contact_steps_by_object': np.sum([x['contact_steps'] for x in summaries], axis=0).tolist(),
               'seconds': round(time.time() - begin, 1), 'rows': summaries}
    (output / f'{split}_data.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps({key: value for key, value in summary.items() if key != 'rows'}), flush=True)
    return summary


class ContactEpisodes:
    """Episode-aware command data; no windows cross from one scene into another."""

    def __init__(self, folder, horizon=3):
        features, actions, contacts = [], [], []
        self.seeds, starts = [], []
        offset = 0
        for path in sorted(Path(folder).glob('*.npz')):
            with np.load(path) as data:
                feature = data['features'].astype(np.float32)
                action = data['actions'].astype(np.float32)
                force = data['contact_force_n'].astype(np.float32)
                if len(feature) != len(action) + 1 or len(force) != len(feature):
                    raise ValueError(f'Observation, action and contact counts do not match in {path}.')
                if not np.allclose(np.diff(data['times']), action[:, 4], atol=1e-6):
                    raise ValueError(f'Command duration does not match the stored times in {path}.')
                self.seeds.append(int(data['seed']))
            labels = (force > .01).astype(np.float32)
            if len(feature) > horizon + 2:
                starts.extend(range(offset + 2, offset + len(feature) - horizon))
            features.append(feature)
            actions.append(np.concatenate([action, np.zeros((1, 5), np.float32)]))
            contacts.append(labels)
            offset += len(feature)
        if not starts:
            raise ValueError(f'No contact-prediction windows were found in {folder}.')
        self.features = torch.from_numpy(np.concatenate(features))
        self.actions = torch.from_numpy(np.concatenate(actions))
        self.contacts = torch.from_numpy(np.concatenate(contacts))
        self.starts = torch.tensor(starts, dtype=torch.long)
        self.horizon = horizon

    def batch(self, ids):
        now = self.starts[ids]
        history = now[:, None] + torch.arange(-2, 1)
        future = now[:, None] + torch.arange(1, self.horizon + 1)
        future_history = future[:, :, None] + torch.arange(-2, 1)
        actions = self.actions[now[:, None] + torch.arange(self.horizon)]
        return self.features[history], self.features[future_history], actions, self.contacts[future]


def source_hashes(config):
    paths = [Path(config['source']['checkpoint']), Path('configs/world_contact.json'),
             Path(config['source']['policy']), Path('uv.lock'),
             Path('src/tinyvla/world_contact_data.py'), Path('src/tinyvla/world_contact_train.py'),
             Path('src/tinyvla/world_contact.py'), Path('src/tinyvla/world_model.py'),
             Path('src/tinyvla/world_data.py'), Path('src/tinyvla/sim.py'),
             Path('src/tinyvla/evaluate.py'), Path('src/tinyvla/language.py'),
             Path('src/tinyvla/controller.py')]
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}

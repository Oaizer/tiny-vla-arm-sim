"""Record arm commands and their results in simulation."""
import json
import time
from pathlib import Path

import numpy as np
import pybullet as p
import torch
from torch.nn import functional as F

from .evaluate import load_policy, instruction_for
from .language import OBJECTS, parse
from .sim import ArmSim

FEATURE_SIZE = 537
TARGET_SIZE = 36
XYZ_OFFSET = np.array([.5, 0, .02], np.float32)
XYZ_SCALE = np.array([.3, .3, .25], np.float32)
STATE_OFFSET = np.array([.5, 0, 0, 0], np.float32)
STATE_SCALE = np.array([.3, .3, .3, .04], np.float32)


class VisualFeatures:
    """Read camera and sensor inputs with the existing fixed visual network."""
    def __init__(self, policy):
        self.policy = policy.eval()

    @torch.inference_mode()
    def __call__(self, observation):
        rgb = torch.as_tensor(observation['rgb']).permute(2, 0, 1)[None].float() / 255
        # Training and use share the same small image and resize operation.
        rgb = F.interpolate(rgb, (64, 64), mode='area')
        rgb = F.interpolate(rgb, (128, 128), mode='bilinear', align_corners=False)
        hidden = self.policy.vision[:6](rgb)
        logits = self.policy.vision[6](hidden)
        _, _, height, width = logits.shape
        yy, xx = torch.meshgrid(torch.linspace(0, 1, height), torch.linspace(0, 1, width), indexing='ij')
        uv = logits.flatten(2).softmax(-1) @ torch.stack([xx, yy], -1).reshape(-1, 2)
        ray = torch.cat([uv, torch.ones_like(uv[..., :1])], -1)
        calibration = torch.as_tensor(observation['calibration'])[None]
        world = torch.einsum('bcij,bcj->bci', calibration, ray)
        world = world[..., :2] / world[..., 2:3]
        world = (world - torch.tensor([.5, 0])) / .3
        presence = logits.flatten(2).amax(-1).sigmoid()
        state = torch.as_tensor((observation['state'] - STATE_OFFSET) / STATE_SCALE)
        return torch.cat([F.adaptive_avg_pool2d(hidden, (4, 4)).flatten(), world.flatten(),
                          presence.flatten(), state]).numpy().astype(np.float32)


def training_target(sim):
    """Read simulator labels for training and offline scoring only."""
    state = sim.state()
    result = []
    for name in OBJECTS:
        if name not in sim.objects:
            result.extend([0.] * 8)
            continue
        xyz, rotation = p.getBasePositionAndOrientation(sim.objects[name])
        xyz = np.asarray(xyz)
        up = np.asarray(p.getMatrixFromQuaternion(rotation)).reshape(3, 3)[:, 2]
        held = xyz[2] > .045 and np.linalg.norm(xyz - state[:3]) < .07 and state[3] > .008
        result.extend([*((xyz - XYZ_OFFSET) / XYZ_SCALE), *up, 1., float(held)])
    return np.array([*result, *((state - STATE_OFFSET) / STATE_SCALE)], np.float32)


def tip_object(sim, task):
    """Set up a fallen object before a training or test trial."""
    if task.object not in sim.objects:
        return
    body = sim.objects[task.object]
    xyz, _ = p.getBasePositionAndOrientation(body)
    rotation = p.getQuaternionFromEuler([np.pi / 2, 0, .35])
    p.resetBasePositionAndOrientation(body, [xyz[0], xyz[1], .028], rotation)
    p.resetBaseVelocity(body, [0, 0, 0], [0, 0, 0])
    sim.step(100)


class RecordingSim(ArmSim):
    def __init__(self, extractor):
        self.recording = False
        self.extractor = extractor
        super().__init__(64)

    def capture(self):
        obs = self.observation()
        self.images.append(obs['rgb'])
        self.features.append(self.extractor(obs))
        self.targets.append(training_target(self))
        self.times.append(self.ticks / 240)

    def begin(self):
        self.images, self.features, self.targets, self.actions, self.times = [], [], [], [], []
        self.capture()
        self.recording = True

    def move(self, xyz, grip, steps=120, frames=None):
        xyz = np.clip(np.array(xyz), [.28, -.33, .015], [.78, .37, .38])
        grip = float(np.clip(grip, 0, .04))
        super().move(xyz, grip, steps, frames)
        if self.recording:
            self.actions.append([*xyz, grip, steps / 240])
            self.capture()


def collect(config, root, policy_path, split, limit=None, offset=0):
    root = Path(root)
    folder = root / 'trajectories' / split
    folder.mkdir(parents=True, exist_ok=True)
    settings = config['data']
    available = settings[f'{split}_episodes'] - offset
    count = available if limit is None else min(limit, available)
    start = settings[f'{split}_seed']
    policy, _ = load_policy(policy_path)
    extractor = VisualFeatures(policy)
    sim = RecordingSim(extractor)
    begin = time.time()
    rows = []
    try:
        for index in range(offset, offset + count):
            seed = start + index
            path = folder / f'{seed}.npz'
            meta_path = path.with_suffix('.json')
            if path.exists() and meta_path.exists():
                rows.append(json.loads(meta_path.read_text()))
                continue
            rng = np.random.default_rng(seed)
            task = parse(instruction_for(index))
            sim.recording = False
            sim.reset(seed)
            tipped = bool('cylinder' in task.object and rng.random() < .65)
            if tipped:
                tip_object(sim, task)
            sim.begin()
            dropped = False
            for attempt in range(2):
                commands = sim.teacher(task)
                # Poor approaches and varied contact heights supply failures as well as successes.
                error = rng.normal(0, .016 if rng.random() < .45 else .004, 2)
                commands[:4, :2] += error
                commands[1:3, 2] = rng.uniform(.015, .047)
                for phase, command in enumerate(commands):
                    duration = int(rng.choice([120, 150, 190]))
                    if phase in (3, 4, 5):
                        sim.move_smooth(command[:3], command[3], duration)
                    else:
                        sim.move(command[:3], command[3], duration)
                    if phase == 3 and attempt == 0 and rng.random() < .30:
                        # These are recorded, known gripper commands, not hidden state changes.
                        sim.move(sim.state()[:3], .04, 80)
                        sim.move(sim.state()[:3], 0, 60)
                        dropped = True
                if sim.score(task)['success']:
                    break
                sim.move(sim.park, .04, 180)
            sim.recording = False
            metadata = {'seed': seed, 'instruction': task.instruction, 'split': split,
                        'transitions': len(sim.actions), 'tipped': tipped, 'drop_commands': dropped,
                        'success': sim.score(task)['success']}
            np.savez_compressed(path, images=np.array(sim.images), features=np.array(sim.features, np.float16),
                                targets=np.array(sim.targets), actions=np.array(sim.actions, np.float32),
                                times=np.array(sim.times, np.float64), seed=seed)
            meta_path.write_text(json.dumps(metadata, indent=2))
            rows.append(metadata)
            if (index + 1) % 25 == 0:
                print(json.dumps({'split': split, 'episodes': index + 1, 'seconds': round(time.time() - begin, 1)}), flush=True)
    finally:
        sim.close()
    result = {'split': split, 'episodes': len(rows), 'transitions': sum(r['transitions'] for r in rows),
              'seed_start': start + offset, 'seed_end': start + offset + count - 1, 'rows': rows}
    suffix = f'_{offset}' if offset else ''
    (root / f'{split}_data{suffix}.json').write_text(json.dumps(result, indent=2))
    return result

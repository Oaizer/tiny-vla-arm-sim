"""Compare short grasp plans using camera features and arm sensors."""
import time
from collections import deque

import numpy as np
import torch

from .controller import Agent
from .language import OBJECTS, parse
from .world_data import XYZ_OFFSET, XYZ_SCALE, STATE_OFFSET, STATE_SCALE
from .world_model import pad_history


class GraspPlanner:
    def __init__(self, model, settings):
        self.model = model.eval()
        self.settings = settings
        self.decisions = []
        offsets = [(x, y, z) for z in [-.012, -.006, 0, .006, .012]
                   for x, y in [(0, 0), (-.006, 0), (.006, 0), (0, -.006), (0, .006)]]
        self.offsets = np.array([(0, 0, 0)] + [v for v in offsets if v != (0, 0, 0)], np.float32)

    @torch.inference_mode()
    def choose(self, history, xyz, grip, steps, source):
        begin = time.perf_counter()
        offsets = self.offsets[:self.settings['candidates']]
        first = np.clip(np.asarray(xyz)[None] + offsets, [.28, -.33, .015], [.78, .37, .38])
        actions = np.zeros((len(first), 3, 5), np.float32)
        actions[:, 0, :3] = first
        actions[:, 0, 3:] = [grip, steps / 240]
        if grip > .02:
            actions[:, 1, :3] = first
            actions[:, 1, 3:] = [0, 190 / 240]
            actions[:, 2, :3] = first + [0, 0, .03]
            actions[:, 2, 3:] = [0, 24 / 240]
        else:
            for step in (1, 2):
                actions[:, step, :3] = first + [0, 0, .025 * step]
                actions[:, step, 3:] = [0, 24 / 240]
        latent = self.model.encode(torch.from_numpy(pad_history(history))[None])
        _, predictions = self.model.rollout(latent.expand(len(first), -1), torch.from_numpy(actions))
        final = predictions[:, :, -1]
        objects = final[..., source * 8:source * 8 + 8]
        position = objects[..., :3] * torch.tensor(XYZ_SCALE) + torch.tensor(XYZ_OFFSET)
        gripper = final[..., 32:35] * torch.tensor(STATE_SCALE[:3]) + torch.tensor(STATE_OFFSET[:3])
        held = objects[..., 7].clamp(0, 1)
        distance = (position - gripper).norm(dim=-1) / .1
        height = ((position[..., 2] - .02) / .1).clamp(0, 1)
        costs = distance - 2 * held - height + 2 * (1 - objects[..., 6].clamp(0, 1))
        disagreement = costs.std(0)
        change = torch.tensor(np.linalg.norm(first - np.asarray(xyz), axis=1) / .012)
        scores = costs.mean(0) + self.settings['uncertainty_weight'] * disagreement + self.settings['change_penalty'] * change
        best = int(scores.argmin())
        gain = float(scores[0] - scores[best])
        accepted = (best != 0 and gain >= self.settings['minimum_gain']
                    and float(disagreement[best]) <= self.settings['max_disagreement'])
        choice = best if accepted else 0
        record = {'accepted': accepted, 'candidate': choice, 'predicted_gain': gain,
                  'disagreement': float(disagreement[best]), 'offset_m': (first[choice] - np.asarray(xyz)).tolist(),
                  'milliseconds': (time.perf_counter() - begin) * 1000}
        self.decisions.append(record)
        return first[choice]


class SensorArm:
    """Give the planner camera and sensor access through a small arm interface."""
    def __init__(self, arm, extractor, planner=None):
        self._arm = arm
        self.extractor = extractor
        self.planner = planner
        self.park = arm.park
        self.history = deque(maxlen=3)
        self.feature_ms = []
        self.source = None
        self.source_xy = None
        self.planning_enabled = not (planner and getattr(planner, 'settings', {}).get('recovery_only', False))
        self.refresh()

    def refresh(self):
        start = time.perf_counter()
        features = self.extractor(self._arm.observation(size=64))
        if not self.history or not np.array_equal(self.history[-1], features):
            self.history.append(features)
        self.feature_ms.append((time.perf_counter() - start) * 1000)

    def observation(self):
        return self._arm.observation()

    def state(self):
        return self._arm.state()

    def move(self, xyz, grip, steps=120, frames=None):
        xyz = np.asarray(xyz).copy()
        near_source = self.source_xy is not None and np.linalg.norm(xyz[:2] - self.source_xy) < .065
        if self.planner is not None and self.planning_enabled and self.source is not None and near_source and xyz[2] < .07:
            xyz = self.planner.choose(self.history, xyz, grip, steps, self.source)
        self._arm.move(xyz, grip, steps, frames)
        self.refresh()

    def move_smooth(self, xyz, grip, steps=150, frames=None):
        start = self.state()[:3]
        goal = np.array(xyz, float)
        pieces = max(1, int(np.ceil(np.linalg.norm(goal - start) / .025)))
        substeps = max(24, int(np.ceil(steps / pieces)))
        for fraction in np.linspace(1 / pieces, 1, pieces):
            self.move(start + (goal - start) * fraction, grip, substeps, frames)

    def step(self, count, frames=None):
        self._arm.step(count, frames)
        self.refresh()


class PlanningAgent(Agent):
    def __init__(self, policy, instruction, settings):
        super().__init__(policy, instruction, settings)
        self.source = OBJECTS.index(parse(instruction).object)

    def observe(self, sim, phase):
        actions, info = super().observe(sim, phase)
        sim.refresh()
        if phase == 'recovery':
            sim.planning_enabled = True
        if phase in ('initial', 'recovery', 'pregrasp_replan'):
            sim.source = self.source
            sim.source_xy = actions[1, :2].copy()
        return actions, info

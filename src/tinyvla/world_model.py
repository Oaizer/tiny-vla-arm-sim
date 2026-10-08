"""Predict future camera features and arm outcomes from proposed commands."""
import copy

import numpy as np
import torch
from torch import nn

from .world_data import FEATURE_SIZE, TARGET_SIZE, STATE_OFFSET, STATE_SCALE


class OutcomeReadout(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.residual = nn.Sequential(nn.Linear(size, 128), nn.GELU(), nn.Linear(128, TARGET_SIZE))
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, latent):
        result = self.residual(latent)
        base = torch.zeros_like(result)
        for obj in range(4):
            base[..., obj * 8:obj * 8 + 2] = latent[..., obj * 2:obj * 2 + 2]
            base[..., obj * 8 + 5] = 1
            base[..., obj * 8 + 6] = latent[..., 14 + obj]
        base[..., 32:] = latent[..., 21:25]
        return base + result


class WorldModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        settings = config['training']
        self.history = config['data']['history']
        size, hidden = settings['latent_size'], settings['hidden_size']
        self.register_buffer('mean', torch.zeros(FEATURE_SIZE))
        self.register_buffer('std', torch.ones(FEATURE_SIZE))
        self.encoder = nn.Sequential(nn.Linear(FEATURE_SIZE * self.history, hidden), nn.LayerNorm(hidden),
                                     nn.GELU(), nn.Linear(hidden, size - 25), nn.LayerNorm(size - 25))
        self.target_encoder = copy.deepcopy(self.encoder)
        self.target_encoder.requires_grad_(False)
        self.predictors = nn.ModuleList([nn.GRUCell(5, size) for _ in range(3)])
        self.readout = OutcomeReadout(size)

    def encode(self, history, target=False):
        value = ((history - self.mean) / self.std).clamp(-8, 8).flatten(1)
        learned = (self.target_encoder if target else self.encoder)(value)
        # Keep measured sensors and visual landmarks as explicit latent coordinates.
        return torch.cat([history[:, -1, 512:], learned], -1)

    def normalize_actions(self, actions):
        offset = actions.new_tensor([*STATE_OFFSET, 0])
        scale = actions.new_tensor([*STATE_SCALE, 1])
        return (actions - offset) / scale

    def rollout(self, latent, actions):
        actions = self.normalize_actions(actions)
        predictions = []
        for cell in self.predictors:
            state = latent
            steps = []
            for action in actions.unbind(1):
                state = cell(action, state)
                steps.append(state)
            predictions.append(torch.stack(steps, 1))
        states = torch.stack(predictions)
        return states, self.readout(states)

    @torch.no_grad()
    def update_target(self, decay):
        for target, online in zip(self.target_encoder.parameters(), self.encoder.parameters()):
            target.mul_(decay).add_(online, alpha=1 - decay)


def load_world(path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    model = WorldModel(checkpoint['config'])
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model, checkpoint


def pad_history(frames, count=3):
    recent = list(frames)[-count:]
    return np.stack([recent[0]] * (count - len(recent)) + recent)

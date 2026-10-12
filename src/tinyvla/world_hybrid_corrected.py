"""Guarded recovery planner with corrected contact feature channels."""
from __future__ import annotations

import time

import numpy as np
import torch

from .language import OBJECTS
from .world_contact_rank import candidates
from .world_contact_reranker import Ranker
from .world_contact_train import ContactHead, OBJECTS as CONTACT_OBJECTS
from .world_data import STATE_SCALE
from .world_model import pad_history


class CorrectedHybridGraspPlanner:
    """Propose a small offset; the feedback controller verifies every grasp."""

    def __init__(self, world_model, settings, contact_path, ranker_path):
        self.model = world_model.eval()
        self.settings = settings
        self.decisions = []
        contact = torch.load(contact_path, map_location='cpu', weights_only=False)
        self.contact = ContactHead(contact['latent_size'], contact['hidden_size'])
        self.contact.load_state_dict(contact['state_dict']); self.contact.eval()
        rank = torch.load(ranker_path, map_location='cpu', weights_only=False)
        self.ranker = Ranker(len(rank['feature_names']), rank['hidden_size'])
        self.ranker.load_state_dict(rank['state_dict']); self.ranker.eval()
        self.mean = torch.as_tensor(rank['mean'], dtype=torch.float32)
        self.std = torch.as_tensor(rank['std'], dtype=torch.float32)
        self.red_state_channel = OBJECTS.index('red cylinder')
        self.red_contact_channel = CONTACT_OBJECTS.index('red cylinder')

    @torch.inference_mode()
    def choose(self, history, xyz, grip, steps, source):
        started = time.perf_counter()
        anchor = np.asarray(xyz, dtype=np.float32)
        if source != self.red_state_channel:
            self.decisions.append({'accepted': False, 'reason': 'unsupported_object',
                                   'object_index': int(source), 'milliseconds': (time.perf_counter()-started)*1000})
            return anchor
        points = candidates(anchor)
        sequence = np.zeros((len(points), 3, 5), np.float32)
        sequence[:, 0, :3], sequence[:, 0, 3:] = points, [.04, 150/240]
        sequence[:, 1, :3], sequence[:, 1, 3:] = points, [0, 190/240]
        sequence[:, 2, :3], sequence[:, 2, 3:] = points + [0, 0, .03], [0, 24/240]
        latent = self.model.encode(torch.from_numpy(pad_history(history))[None]).expand(len(points), -1)
        states, outcomes = self.model.rollout(latent, torch.from_numpy(sequence))
        logits = self.contact(states.permute(1, 0, 2, 3)).permute(1, 0, 2, 3)
        contact_prob = logits[:, :, 1, self.red_contact_channel].sigmoid().T.numpy()
        disagreement = contact_prob.std(axis=1)
        channel = self.red_state_channel
        held = outcomes[:, :, 1, channel*8+7].T.numpy().clip(0, 1)
        target = outcomes[:, :, 1, channel*8:channel*8+3].mean(0).numpy()
        target = target * STATE_SCALE[:3] + np.array([0., 0., 0.])
        # State outputs in this feature path are normalized against fixed offsets.
        from .world_data import STATE_OFFSET
        target = target + STATE_OFFSET[:3]
        features = np.concatenate([contact_prob, held, points-anchor, points-target,
                                   np.zeros((len(points), 1), np.float32)], axis=1).astype(np.float32)
        scores = self.ranker((torch.from_numpy(features)-self.mean)/self.std).numpy()
        best = int(scores.argmax()); margin = float(scores[best]-scores[0])
        uncertainty = float(disagreement[best])
        accepted = best != 0 and margin >= float(self.settings.get('hybrid_min_score_gain', .05)) \
            and uncertainty <= float(self.settings.get('hybrid_max_contact_disagreement', .25))
        chosen = best if accepted else 0
        self.decisions.append({'accepted': accepted, 'candidate': chosen, 'proposed_candidate': best,
                               'score_margin': margin, 'contact_disagreement': uncertainty,
                               'predicted_contact': float(contact_prob[chosen].mean()),
                               'offset_m': (points[chosen]-anchor).tolist(), 'fallback': not accepted,
                               'milliseconds': (time.perf_counter()-started)*1000})
        return points[chosen]

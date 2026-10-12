"""Allow one bounded recovery from a temporarily hidden source object."""
from __future__ import annotations

from .world_planner import PlanningAgent


class RecoveryFallbackAgent(PlanningAgent):
    """Reuse the last valid visual plan only after a failed grasp."""

    def __init__(self, policy, instruction, settings):
        super().__init__(policy, instruction, settings)
        self.last_good_action = None
        self.current_phase = None
        self.stale_fallback = False

    def observe(self, sim, phase):
        actions, info = super().observe(sim, phase)
        self.current_phase = phase
        threshold = self.settings['presence_threshold']
        if info['source_confidence'] >= threshold:
            if phase in ('initial', 'recovery', 'pregrasp_replan'):
                self.last_good_action = actions.copy()
            self.stale_fallback = False
            return actions, info
        if phase in ('recovery', 'pregrasp_replan') and self.last_good_action is not None:
            actions = self.last_good_action.copy()
            self.stale_fallback = True
            self.trace.append({'phase': 'stale_source_fallback', 'at_phase': phase,
                               'source_confidence': float(info['source_confidence'])})
            return actions, info
        self.stale_fallback = False
        return actions, info

    def valid(self, action, info, source=True):
        if self.stale_fallback and source and self.current_phase in ('recovery', 'pregrasp_replan'):
            # Keep destination visibility and workspace validation. Only waive
            # a low source score because the source has a recent visual anchor.
            recovered_info = dict(info)
            recovered_info['source_confidence'] = self.settings['presence_threshold']
            return super().valid(action, recovered_info, source)
        return super().valid(action, info, source)

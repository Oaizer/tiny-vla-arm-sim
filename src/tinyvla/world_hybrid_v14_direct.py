"""Direct-approach controller used to remove the initial probe-and-return."""
from __future__ import annotations

from .controller import Agent
from .language import OBJECTS, parse
from .world_planner import PlanningAgent


class DirectApproachAgent(Agent):
    """Use the first camera plan directly instead of probing and retreating."""

    def run(self, sim, mode='feedback', event=None, frames=None):
        retries = 0
        recovered = False
        threshold = self.settings['empty_gripper_threshold_m']
        carry = getattr(sim, 'move_smooth', sim.move)
        try:
            action, info = self.observe(sim, 'initial')
            valid, reason = self.valid(action, info)
            if not valid:
                return self.report('stopped_' + reason, retries, recovered)

            for attempt in range(self.settings['max_attempts']):
                if attempt:
                    retries += 1
                    sim.move(sim.park, .04, 180, frames)
                    action, info = self.observe(sim, 'recovery')
                    valid, reason = self.valid(action, info)
                    if not valid:
                        return self.report('stopped_' + reason, retries, recovered)

                # The initial camera view is already from the clear parked pose.
                # Follow that plan directly; do not move in and retreat to view again.
                sim.move(action[0, :3], action[0, 3], 120, frames)
                if event:
                    event('after_approach')
                sim.move(action[1, :3], action[1, 3], 150, frames)
                sim.move(action[2, :3], action[2, 3], 190, frames)
                carry(action[3, :3], action[3, 3], 160, frames)
                if event:
                    event('after_lift')

                if sim.state()[3] < threshold:
                    self.trace.append({'phase': 'empty_gripper_retry', 'attempt': attempt + 1})
                    recovered = True
                    continue
                if info['task'] == 1:
                    return self.report('lifted', retries, recovered)

                updated, pad_info = self.observe(sim, 'destination_replan')
                valid, reason = self.valid(updated, pad_info, source=False)
                if not valid:
                    return self.report('stopped_' + reason, retries, recovered)
                lost = False
                for index in [4, 5]:
                    carry(updated[index, :3], updated[index, 3], 150, frames)
                    if sim.state()[3] < threshold:
                        lost = True
                        recovered = True
                        self.trace.append({'phase': 'transport_slip_retry', 'attempt': attempt + 1})
                        break
                if lost:
                    continue
                for index in [6, 7]:
                    sim.move(updated[index, :3], updated[index, 3], 170 if index == 6 else 150, frames)
                sim.step(100, frames)
                return self.report('placed', retries, recovered)

            sim.move(sim.park, .04, 150, frames)
            return self.report('stopped_attempt_budget', retries, recovered)
        except (RuntimeError, ValueError) as error:
            return self.report('stopped_invalid_prediction', retries, recovered, str(error))


class DirectApproachPlanningAgent(DirectApproachAgent):
    """Planning-agent behavior with the direct-approach motion sequence."""

    def __init__(self, policy, instruction, settings):
        super().__init__(policy, instruction, settings)
        self.source = OBJECTS.index(parse(instruction).object)

    def observe(self, sim, phase):
        actions, info = super().observe(sim, phase)
        sim.refresh()
        if phase == 'recovery':
            sim.planning_enabled = True
        if phase in ('initial', 'recovery'):
            sim.source = self.source
            sim.source_xy = actions[1, :2].copy()
        return actions, info


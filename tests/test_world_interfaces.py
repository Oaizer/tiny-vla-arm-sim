"""Check action timing and the sensor interface used by the world planner."""
import unittest

import numpy as np

from tinyvla.sim import ArmSim
from tinyvla.world_data import RecordingSim, FEATURE_SIZE
from tinyvla.world_planner import SensorArm


def simple_features(observation):
    result = np.zeros(FEATURE_SIZE, np.float32)
    result[-4:] = observation['state']
    return result


class WorldInterfaceTests(unittest.TestCase):
    def test_recorded_times_match_actual_servo_commands(self):
        sim = RecordingSim(simple_features)
        try:
            sim.reset(901)
            sim.begin()
            sim.move([.45, -.1, .2], .04, steps=24)
            sim.move([.46, -.1, .18], .02, steps=48)
            np.testing.assert_allclose(np.diff(sim.times), [.1, .2], atol=1e-8)
            np.testing.assert_allclose(np.asarray(sim.actions)[:, 4], np.diff(sim.times))
            np.testing.assert_allclose(sim.features[-1][-4:], sim.state())
            self.assertEqual(len(sim.images), len(sim.actions) + 1)
        finally:
            sim.close()

    def test_sensor_wrapper_preserves_original_motion(self):
        sim = ArmSim(64)
        try:
            sim.reset(902)
            sim.move([.45, -.1, .2], .04, 120)
            sim.move_smooth([.52, -.08, .10], .02, 150)
            expected = sim.state().copy()
            commands = sim.command_count
            sim.reset(902)
            wrapper = SensorArm(sim, simple_features)
            wrapper.move([.45, -.1, .2], .04, 120)
            wrapper.move_smooth([.52, -.08, .10], .02, 150)
            np.testing.assert_allclose(wrapper.state(), expected, atol=1e-7)
            self.assertEqual(sim.command_count, commands)
        finally:
            sim.close()

    def test_planning_interface_works_without_object_state(self):
        class SensorOnly:
            park = np.array([.27, -.35, .34])
            def __init__(self):
                self.value = np.array([.5, 0, .2, .04], np.float32)
            def observation(self, size=None):
                return {'rgb': np.zeros((64, 64, 3), np.uint8), 'state': self.state(),
                        'calibration': np.zeros((7, 3, 3), np.float32)}
            def state(self):
                return self.value.copy()
            def move(self, xyz, grip, steps, frames):
                self.value = np.array([*xyz, grip], np.float32)
            def step(self, count, frames):
                pass
        class Planner:
            def choose(self, history, xyz, grip, steps, source):
                self.called = True
                return xyz
        planner = Planner()
        wrapper = SensorArm(SensorOnly(), simple_features, planner)
        wrapper.source, wrapper.source_xy = 0, np.array([.5, 0])
        wrapper.move([.5, 0, .026], .04, 150)
        self.assertTrue(planner.called)
        self.assertFalse(hasattr(wrapper, 'objects'))
        self.assertFalse(hasattr(wrapper, 'labels'))


if __name__ == '__main__':
    unittest.main()

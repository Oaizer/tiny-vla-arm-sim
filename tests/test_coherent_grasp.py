"""Check that a planned grasp keeps its position through finger closure."""
import unittest
import numpy as np
from tinyvla.world_planner import SensorArm
from tinyvla.world_data import FEATURE_SIZE


class CoherentGraspTests(unittest.TestCase):
    def test_close_keeps_offset_and_lift_is_unmodified(self):
        class Arm:
            park = np.array([.3, -.3, .3])
            def observation(self, size=None):
                return {}
            def move(self, xyz, grip, steps, frames):
                commands.append(np.array(xyz))
        class Planner:
            settings = {'coherent_grasp': True}
            calls = 0
            def choose(self, history, xyz, grip, steps, source):
                self.calls += 1
                return xyz + np.array([.006, 0, -.006])
        commands = []
        planner = Planner()
        view = SensorArm(Arm(), lambda obs: np.zeros(FEATURE_SIZE), planner)
        view.source, view.source_xy = 0, np.array([.5, 0])
        view.move([.5, 0, .026], .04, 150)
        view.move([.5, 0, .026], 0, 190)
        view.move([.5, 0, .05], 0, 24)
        np.testing.assert_allclose(commands[0], [.506, 0, .020])
        np.testing.assert_allclose(commands[1], commands[0])
        np.testing.assert_allclose(commands[2], [.5, 0, .05])
        self.assertEqual(planner.calls, 1)
        view.move([.5, 0, .026], .04, 150)
        view.move([.3, -.3, .3], .04, 150)
        view.move([.5, 0, .026], 0, 190)
        np.testing.assert_allclose(commands[-1], [.5, 0, .026])
        self.assertEqual(planner.calls, 2)


if __name__ == '__main__':
    unittest.main()

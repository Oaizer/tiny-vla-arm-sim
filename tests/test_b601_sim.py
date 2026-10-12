"""Regression checks for the articulated Seeed B601-RS simulation pilot."""
import unittest

from tinyvla.sim_b601 import B601Sim, LIMITS, pb


class B601SimulationContracts(unittest.TestCase):
    def test_seeded_teacher_trial_uses_the_arm_and_places_the_object(self):
        sim = B601Sim()
        try:
            sim.reset(0)
            arm_joint_names = [pb.getJointInfo(sim.arm, joint)[1].decode() for joint in sim.arm_joints]
            self.assertEqual(arm_joint_names, [f"joint{i}" for i in range(1, 7)])
            outcome = sim.teacher_trial("red cube", "yellow")
            self.assertTrue(outcome["success"])
            self.assertTrue(outcome["joint_limits_respected"])
            self.assertEqual(outcome["commands"], 8)
            self.assertLess(outcome["distance_m"], .055)
            for joint, (lower, upper) in zip(sim.arm_joints, LIMITS[:6]):
                position = pb.getJointState(sim.arm, joint)[0]
                self.assertGreaterEqual(position, lower - 1e-5)
                self.assertLessEqual(position, upper + 1e-5)
        finally:
            sim.close()


if __name__ == "__main__":
    unittest.main()

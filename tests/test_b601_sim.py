"""Regression checks for the articulated Seeed B601-RS simulation pilot."""
import unittest

from tinyvla.sim_b601 import B601Sim, LIMITS, pb


class B601SimulationContracts(unittest.TestCase):
    def test_dataset_observer_records_only_camera_proprioception_and_waypoint(self):
        sim = B601Sim(size=128, observation_size=128)
        try:
            sim.reset(2)
            records = []
            sim.current_instruction = "sort the red cube onto the yellow pad"
            sim.command_observer = lambda observation, action: records.append((observation, action))
            sim._observe_command(sim.q.copy(), [.05, .0715])
            observation, action = records[0]
            self.assertEqual(observation["image"].shape, (128, 128, 3))
            self.assertEqual(observation["joint_position"].shape, (6,))
            self.assertEqual(observation["joint_velocity"].shape, (6,))
            self.assertEqual(observation["gripper_position"].shape, (2,))
            self.assertEqual(action.shape, (8,))
            self.assertEqual(observation["instruction"], sim.current_instruction)
            self.assertEqual(observation["phase"], -1)
            self.assertNotIn("object_position", observation)
            self.assertNotIn("pad_position", observation)
        finally:
            sim.close()

    def test_folded_joint_move_respects_the_reference_speed_cap(self):
        sim = B601Sim()
        try:
            sim.reset(1)
            target = sim.home_q.copy()
            target[0] = .2
            sim.move_joints(target)
            self.assertLessEqual(sim.peak_reference_speed_rad_s, 1.5 + 1e-6)
            self.assertEqual(sim.command_count, 1)
            for value, (lower, upper) in zip(sim.q, LIMITS[:6]):
                self.assertGreaterEqual(value, lower - 1e-5)
                self.assertLessEqual(value, upper + 1e-5)
        finally:
            sim.close()

    def test_waypoint_move_accepts_bounded_finger_targets(self):
        sim = B601Sim()
        try:
            sim.reset(3)
            sim.move_joints(sim.q.copy(), finger_targets=[.02, .03])
            fingers = [pb.getJointState(sim.arm, joint)[0] for joint in sim.gripper_joints]
            self.assertAlmostEqual(fingers[0], .02, delta=.002)
            self.assertAlmostEqual(fingers[1], .03, delta=.002)
            with self.assertRaises(ValueError):
                sim.move_joints(sim.q.copy(), finger_targets=[.06, .03])
        finally:
            sim.close()

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
            self.assertLessEqual(sim.peak_reference_speed_rad_s, 1.5 + 1e-6)
            self.assertLess(outcome["distance_m"], .055)
            for joint, (lower, upper) in zip(sim.arm_joints, LIMITS[:6]):
                position = pb.getJointState(sim.arm, joint)[0]
                self.assertGreaterEqual(position, lower - 1e-5)
                self.assertLessEqual(position, upper + 1e-5)
        finally:
            sim.close()

    def test_phase_teacher_targets_stay_inside_b601_joint_limits(self):
        sim = B601Sim()
        try:
            sim.reset(7)
            for phase in range(8):
                target = sim.teacher_waypoint("red cube", "yellow", phase)
                self.assertEqual(target.shape, (8,))
                for value, (lower, upper) in zip(target, LIMITS):
                    self.assertGreaterEqual(value, lower - 1e-6)
                    self.assertLessEqual(value, upper + 1e-6)
            with self.assertRaises(ValueError):
                sim.teacher_waypoint("red cube", "yellow", 8)
        finally:
            sim.close()


if __name__ == "__main__":
    unittest.main()

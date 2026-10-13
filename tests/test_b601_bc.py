"""Contract checks for the compact B601 behavior-cloning baseline."""
import unittest

import numpy as np
import torch

from tinyvla.b601_bc import (TinyB601Policy, denormalize_actions, encode_instruction,
                             normalize_actions, denormalize_cartesian_actions,
                             normalize_cartesian_actions, joint_targets_to_cartesian,
                             encode_visual_targets)
from tinyvla.sim_b601 import B601Sim, pb


class B601BehaviorCloningContracts(unittest.TestCase):
    def test_action_scaling_round_trips_b601_waypoint_limits(self):
        action = np.asarray([0, .7, .8, -.4, .2, 1.0, .025, .03575], dtype=np.float32)
        np.testing.assert_allclose(denormalize_actions(normalize_actions(action)), action, atol=1e-6)
        cartesian = np.asarray([.31, -.08, .14, -.7, .045], dtype=np.float32)
        np.testing.assert_allclose(denormalize_cartesian_actions(
            normalize_cartesian_actions(cartesian[None]))[0], cartesian, atol=1e-6)

    def test_teacher_joint_waypoints_convert_to_cartesian_actions(self):
        action = np.asarray([[.2, 1.8, 1.7, -1.4, 0.0, -.5, .0185, .0265]], dtype=np.float32)
        cartesian = joint_targets_to_cartesian(action)
        self.assertEqual(cartesian.shape, (1, 5))
        self.assertTrue(np.isfinite(cartesian).all())
        self.assertGreaterEqual(float(cartesian[0, 4]), 0)
        self.assertLessEqual(float(cartesian[0, 4]), .1215)

    def test_instruction_and_policy_shapes(self):
        tokens = encode_instruction("sort the red cube onto the yellow pad")
        self.assertEqual(tokens.shape, (8,))
        self.assertNotEqual(tokens[2], encode_instruction("sort the blue cube onto the purple pad")[2])
        model = TinyB601Policy()
        batch_tokens = torch.from_numpy(np.stack([tokens, tokens]))
        images = torch.zeros(2, 3, 96, 96)
        state = torch.zeros(2, 14)
        output = model(images, state, batch_tokens, torch.tensor([0, 7]))
        self.assertEqual(tuple(output.shape), (2, 8))
        self.assertLess(sum(parameter.numel() for parameter in model.parameters()), 500_000)
        old_policy = TinyB601Policy(spatial_size=4)
        old_output = old_policy(torch.zeros(1, 3, 96, 96), torch.zeros(1, 14),
                                torch.from_numpy(tokens).unsqueeze(0), torch.tensor([0]))
        self.assertEqual(tuple(old_output.shape), (1, 8))
        cartesian_policy = TinyB601Policy(spatial_size=4, action_dim=6)
        cartesian_output = cartesian_policy(torch.zeros(1, 3, 96, 96), torch.zeros(1, 14),
                                            torch.from_numpy(tokens).unsqueeze(0), torch.tensor([2]))
        self.assertEqual(tuple(cartesian_output.shape), (1, 6))

    def test_visual_target_features_read_requested_cube_and_pad(self):
        image = np.zeros((1, 32, 32, 3), dtype=np.uint8)
        image[0, 7:13, 5:11] = [205, 18, 15]
        image[0, 20:28, 21:29] = [218, 171, 9]
        tokens = encode_instruction("sort the red cube onto the yellow pad")[None]
        features = encode_visual_targets(image, tokens)
        self.assertEqual(features.shape, (1, 8))
        self.assertAlmostEqual(float(features[0, 0]), 2 * 7.5 / 31 - 1, places=5)
        self.assertAlmostEqual(float(features[0, 1]), 2 * 9.5 / 31 - 1, places=5)
        self.assertAlmostEqual(float(features[0, 4]), 2 * 24.5 / 31 - 1, places=5)
        self.assertAlmostEqual(float(features[0, 5]), 2 * 23.5 / 31 - 1, places=5)
        self.assertEqual(tuple(features[0, 6:]), (1.0, 1.0))

        green_scene = np.zeros((1, 32, 32, 3), dtype=np.uint8)
        green_scene[0, 5:11, 3:9] = [18, 155, 37]
        green_scene[0, 19:27, 21:29] = [4, 162, 176]
        green_tokens = encode_instruction("sort the green cube onto the cyan pad")[None]
        green_features = encode_visual_targets(green_scene, green_tokens)
        self.assertAlmostEqual(float(green_features[0, 4]), 2 * 24.5 / 31 - 1, places=5)

    def test_overhead_visual_square_yaw_matches_world_yaw(self):
        sim = B601Sim(size=384, observation_size=384, camera_view="overhead")
        try:
            sim.reset(1)
            tokens = encode_instruction("sort the red cube onto the yellow pad")
            for degrees in (-35, -20, 0, 20, 35):
                quat = pb.getQuaternionFromEuler([0, 0, np.deg2rad(degrees)])
                pb.resetBasePositionAndOrientation(sim.objects["red cube"], [.24, -.16, .125], quat)
                features = encode_visual_targets(sim.image(384)[None], tokens[None])[0]
                measured = np.arctan2(features[2], features[3]) / 4
                error = (measured - np.deg2rad(degrees) + np.pi / 4) % (np.pi / 2) - np.pi / 4
                self.assertLess(abs(float(error)), np.deg2rad(3), f"wrong yaw at {degrees} degrees")
                legacy = encode_visual_targets(sim.image(384)[None], tokens[None],
                                                legacy_image_yaw=True)[0]
                legacy_measured = np.arctan2(legacy[2], legacy[3]) / 4
                legacy_error = (legacy_measured + np.deg2rad(degrees) + np.pi / 4) % (np.pi / 2) - np.pi / 4
                self.assertLess(abs(float(legacy_error)), np.deg2rad(3),
                                f"legacy yaw convention changed at {degrees} degrees")
        finally:
            sim.close()

    def test_visual_feature_policy_keeps_compact_output_shape(self):
        model = TinyB601Policy(spatial_size=4, action_dim=6, visual_feature_dim=8)
        output = model(torch.zeros(1, 3, 96, 96), torch.zeros(1, 14),
                       torch.from_numpy(encode_instruction("sort the blue cube onto the purple pad")).unsqueeze(0),
                       torch.tensor([3]), torch.zeros(1, 8))
        self.assertEqual(tuple(output.shape), (1, 6))
        self.assertLess(sum(parameter.numel() for parameter in model.parameters()), 500_000)


if __name__ == "__main__":
    unittest.main()

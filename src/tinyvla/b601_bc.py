"""Small image, instruction, and proprioception behavior-cloning baseline."""
import json
import atexit
from pathlib import Path

import numpy as np
import torch
from torch import nn


JOINT_LIMITS = np.asarray([
    [-2.8, 2.8], [0.0, 3.14], [0.0, 3.14], [-1.57, 1.57],
    [-1.57, 1.57], [-3.14, 3.14], [0.0, .05], [0.0, .0715],
], dtype=np.float32)
VOCAB = {token: index + 1 for index, token in enumerate(
    ("sort", "the", "red", "blue", "green", "cube", "onto", "yellow", "purple", "cyan", "pad"))}
PAD_TOKEN = 0
TEXT_LENGTH = 8
_FK_SIM = None
CARTESIAN_LIMITS = np.asarray([
    [.15, .48], [-.25, .25], [.06, .22], [-np.pi, np.pi], [0, .1215],
], dtype=np.float32)


def encode_instruction(text: str) -> np.ndarray:
    tokens = [VOCAB[token] for token in text.lower().split() if token in VOCAB]
    tokens = tokens[:TEXT_LENGTH]
    return np.asarray(tokens + [PAD_TOKEN] * (TEXT_LENGTH - len(tokens)), dtype=np.int64)


def encode_visual_targets(images: np.ndarray, tokens: np.ndarray) -> np.ndarray:
    """Read target cube and pad centers from color pixels; no simulator poses are used."""
    images = np.asarray(images, dtype=np.uint8)
    tokens = np.asarray(tokens)
    if images.ndim != 4 or images.shape[-1] != 3 or tokens.shape[0] != images.shape[0]:
        raise ValueError("images must be NHW3 and tokens must have the same batch size")
    height, width = images.shape[1:3]
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    xx = 2 * xx / max(width - 1, 1) - 1
    yy = 2 * yy / max(height - 1, 1) - 1
    color_tokens = {color: VOCAB[color] for color in ("red", "blue", "green", "yellow", "purple", "cyan")}
    values = np.zeros((len(images), 8), dtype=np.float32)
    for index, image in enumerate(images):
        present = set(int(value) for value in tokens[index] if int(value))
        object_color = next((color for color in ("red", "blue", "green")
                             if color_tokens[color] in present), None)
        pad_color = next((color for color in ("yellow", "purple", "cyan")
                          if color_tokens[color] in present), None)
        if object_color is None or pad_color is None:
            raise ValueError("instruction must name one cube color and one pad color")
        red, green, blue = [image[..., channel].astype(np.float32) for channel in range(3)]
        masks = {
            "red": (red > 100) & (red > 2.4 * green) & (red > 2.4 * blue),
            "blue": (blue > 100) & (blue > 2.2 * red) & (blue > 2.0 * green),
            "green": (green > 100) & (green > 2.2 * red) & (green > 2.0 * blue),
            "yellow": (red > 100) & (green > 65) & (red > 1.15 * green) & (green > 2.0 * blue),
            "purple": (red > 65) & (blue > 65) & (red > 2.0 * green) & (blue > 2.0 * green),
            "cyan": (green > 65) & (blue > 65) & (green > 1.8 * red) & (blue > 1.8 * red),
        }

        def center(mask):
            count = int(mask.sum())
            if count < 8:
                return 0.0, 0.0, False
            return float(xx[mask].mean()), float(yy[mask].mean()), True

        object_mask = masks[object_color]
        object_x, object_y, object_found = center(object_mask)
        pad_x, pad_y, pad_found = center(masks[pad_color])
        if object_found:
            dx = xx[object_mask] - object_x
            dy = yy[object_mask] - object_y
            z = dx + 1j * dy
            moment = -np.mean(z ** 4)
            magnitude = abs(moment)
            if magnitude > 1e-8:
                orientation = moment / magnitude
                sin_four_yaw, cos_four_yaw = float(orientation.imag), float(orientation.real)
            else:
                sin_four_yaw, cos_four_yaw = 0.0, 0.0
        else:
            sin_four_yaw, cos_four_yaw = 0.0, 0.0
        values[index] = [object_x, object_y, sin_four_yaw, cos_four_yaw,
                         pad_x, pad_y, float(object_found), float(pad_found)]
    return values


def normalize_actions(actions: np.ndarray) -> np.ndarray:
    low, high = JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1]
    return np.clip(2 * (actions - low) / (high - low) - 1, -1, 1)


def denormalize_actions(actions: np.ndarray) -> np.ndarray:
    low, high = JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1]
    return low + (actions + 1) * .5 * (high - low)


def normalize_cartesian_actions(actions: np.ndarray) -> np.ndarray:
    xyz_low, xyz_high = CARTESIAN_LIMITS[:3, 0], CARTESIAN_LIMITS[:3, 1]
    xyz = np.clip(2 * (actions[:, :3] - xyz_low) / (xyz_high - xyz_low) - 1, -1, 1)
    width = np.clip(2 * actions[:, 4:5] / CARTESIAN_LIMITS[4, 1] - 1, -1, 1)
    angle = actions[:, 3]
    return np.concatenate((xyz, np.sin(angle)[:, None], np.cos(angle)[:, None], width), axis=1)


def denormalize_cartesian_actions(actions: np.ndarray) -> np.ndarray:
    xyz_low, xyz_high = CARTESIAN_LIMITS[:3, 0], CARTESIAN_LIMITS[:3, 1]
    xyz = xyz_low + (actions[..., :3] + 1) * .5 * (xyz_high - xyz_low)
    yaw = np.arctan2(actions[..., 3], actions[..., 4])[..., None]
    width = (actions[..., 5:6] + 1) * .5 * CARTESIAN_LIMITS[4, 1]
    return np.concatenate((xyz, yaw, width), axis=-1)


def joint_targets_to_cartesian(actions: np.ndarray) -> np.ndarray:
    """Convert teacher joint waypoints to TCP xyz, wrist yaw, and gripper width."""
    from tinyvla.sim_b601 import B601Sim, pb

    global _FK_SIM
    if _FK_SIM is None:
        _FK_SIM = B601Sim(size=64, observation_size=64)
        _FK_SIM.reset(0)
        atexit.register(_FK_SIM.close)
    sim = _FK_SIM
    inverse_top_down = pb.invertTransform([0, 0, 0], sim.top_down)[1]
    output = np.empty((len(actions), 5), dtype=np.float32)
    for index, action in enumerate(actions):
        for joint, value in zip(sim.arm_joints, action[:6]):
            pb.resetJointState(sim.arm, joint, float(value), targetVelocity=0)
        state = pb.getLinkState(sim.arm, sim.tcp_link, computeForwardKinematics=True)
        yaw_delta = pb.multiplyTransforms([0, 0, 0], state[5], [0, 0, 0], inverse_top_down)[1]
        yaw = float(pb.getEulerFromQuaternion(yaw_delta)[2])
        # Cube yaw is equivalent modulo 90 degrees for face alignment.
        yaw = (yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4
        width = float(np.mean([action[6] / .05, action[7] / .0715]) * .1215)
        output[index] = [*state[4], yaw, width]
    return output


def encode_proprioception(joint_position, joint_velocity, gripper_position) -> np.ndarray:
    joint_position = np.asarray(joint_position, dtype=np.float32)
    joint_velocity = np.asarray(joint_velocity, dtype=np.float32)
    gripper_position = np.asarray(gripper_position, dtype=np.float32)
    gripper_position = np.clip(gripper_position, 0, JOINT_LIMITS[6:, 1])
    arm_limits = JOINT_LIMITS[:6]
    normalized_position = 2 * (joint_position - arm_limits[:, 0]) / (arm_limits[:, 1] - arm_limits[:, 0]) - 1
    normalized_velocity = np.clip(joint_velocity, -10, 10) / 10
    normalized_fingers = 2 * gripper_position / np.asarray([.05, .0715], dtype=np.float32) - 1
    return np.concatenate((normalized_position, normalized_velocity, normalized_fingers)).astype(np.float32)


class TinyB601Policy(nn.Module):
    def __init__(self, spatial_size=8, phase_conditioned=True, action_dim=8,
                 visual_feature_dim=0):
        super().__init__()
        self.spatial_size = int(spatial_size)
        self.phase_conditioned = bool(phase_conditioned)
        self.action_dim = int(action_dim)
        self.visual_feature_dim = int(visual_feature_dim)
        self.vision = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 48, 3, stride=2, padding=1), nn.ReLU(),
            # Keep spatial detail so the compact model can locate small cubes
            # instead of averaging away their positions in the camera view.
            nn.AdaptiveAvgPool2d((self.spatial_size, self.spatial_size)), nn.Flatten(),
        )
        self.text_embedding = nn.Embedding(len(VOCAB) + 1, 16, padding_idx=PAD_TOKEN)
        self.head = nn.Sequential(
            nn.Linear(48 * self.spatial_size * self.spatial_size + 14 + 16
                      + (8 if self.phase_conditioned else 0) + self.visual_feature_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, self.action_dim), nn.Tanh(),
        )

    def forward(self, images, proprioception, tokens, phases=None, visual_features=None):
        image_features = self.vision(images)
        text_mask = (tokens != PAD_TOKEN).float().unsqueeze(-1)
        token_features = self.text_embedding(tokens)
        text_features = (token_features * text_mask).sum(dim=1) / text_mask.sum(dim=1).clamp_min(1)
        features = [image_features, proprioception, text_features]
        if self.phase_conditioned:
            phase_features = (torch.zeros((images.shape[0], 8), dtype=images.dtype, device=images.device)
                              if phases is None else
                              torch.nn.functional.one_hot(phases.long(), num_classes=8).to(images.dtype))
            features.append(phase_features)
        if self.visual_feature_dim:
            if visual_features is None or visual_features.shape[-1] != self.visual_feature_dim:
                raise ValueError("this policy requires its configured visual target features")
            features.append(visual_features.to(dtype=images.dtype))
        return self.head(torch.cat(features, dim=1))


def load_samples(root: Path, successful_only: bool = False, action_mode: str = "joint"):
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    splits = {"train": [], "validation": []}
    for episode in manifest["episodes"]:
        if successful_only and not episode["success"]:
            continue
        with np.load(root / episode["file"], allow_pickle=False) as data:
            instructions = [encode_instruction(value) for value in episode["instruction_per_sample"]]
            proprio = np.stack([encode_proprioception(position, velocity, gripper)
                                for position, velocity, gripper in zip(
                                    data["joint_position"], data["joint_velocity"],
                                    data["gripper_position"])])
            actions = data["actions"]
            if action_mode == "cartesian":
                actions = joint_targets_to_cartesian(actions)
                actions = normalize_cartesian_actions(actions)
            elif action_mode == "joint":
                actions = normalize_actions(actions)
            else:
                raise ValueError(f"unknown action mode: {action_mode}")
            split = splits[episode["split"]]
            split.append((data["images"].copy(), proprio.astype(np.float32),
                          np.stack(instructions), data["phases"].astype(np.int64),
                          actions))
    result = {}
    for split, rows in splits.items():
        if not rows:
            raise ValueError(f"dataset has no {split} episodes")
        result[split] = tuple(np.concatenate([row[index] for row in rows], axis=0) for index in range(5))
    return result


def tensors(batch, device):
    images, proprioception, tokens, phases, actions = batch
    return (
        torch.from_numpy(images).permute(0, 3, 1, 2).to(device),
        torch.from_numpy(proprioception).float().to(device),
        torch.from_numpy(tokens).long().to(device),
        torch.from_numpy(phases).long().to(device),
        torch.from_numpy(actions).float().to(device),
    )


def regression_metrics(predicted: np.ndarray, truth: np.ndarray):
    error = np.abs(denormalize_actions(predicted) - denormalize_actions(truth))
    return {"joint_mae_rad": float(error[:, :6].mean()),
            "gripper_joint_mae_m": float(error[:, 6:].mean()),
            "per_joint_mae": error.mean(axis=0).tolist()}

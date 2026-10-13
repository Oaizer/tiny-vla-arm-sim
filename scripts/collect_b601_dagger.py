"""Collect corrective teacher labels on states visited by a learned B601 policy."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from demo_b601_sort import OBJECTS, OBJECT_TO_PAD, PAD_XY, SLOT_SETS, SUCCESS_TOLERANCE_M, randomize_scene  # noqa: E402
from tinyvla.b601_bc import (TinyB601Policy, denormalize_actions, denormalize_cartesian_actions,
                             encode_instruction, encode_proprioception,
                             encode_visual_targets)  # noqa: E402
from tinyvla.sim_b601 import B601Sim, LIMITS, pb  # noqa: E402


def collect_episode(model, sim, seed, action_mode="joint", visual_yaw_convention="legacy",
                    intervention_position_m=.025, intervention_yaw_rad=.25,
                    intervention_gripper_m=.015):
    sim.reset(seed)
    _, rng = randomize_scene(sim, seed, SLOT_SETS["v7"], min_target_distance=.075)
    order = [OBJECTS[index] for index in rng.permutation(len(OBJECTS))]
    images, positions, velocities, grippers, actions, phases, instructions = [], [], [], [], [], [], []
    object_results = []
    teacher_interventions = 0
    model.eval()
    for name in order:
        pad = OBJECT_TO_PAD[name]
        sim.current_instruction = f"sort the {name} onto the {pad} pad"
        source, source_orientation = pb.getBasePositionAndOrientation(sim.objects[name])
        destination = pb.getBasePositionAndOrientation(sim.pads[pad])[0]
        cube_yaw = pb.getEulerFromQuaternion(source_orientation)[2]
        cube_yaw = (cube_yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4
        yaw_rotation = pb.getQuaternionFromEuler([0, 0, float(cube_yaw)])
        orientation = pb.multiplyTransforms([0, 0, 0], yaw_rotation,
                                            [0, 0, 0], sim.top_down)[1]
        waypoints = (
            ((source[0], source[1], .14), 121.5), ((source[0], source[1], .08), 121.5),
            ((source[0], source[1], .08), 45), ((source[0], source[1], .20), 45),
            ((destination[0], destination[1], .20), 45),
            ((destination[0], destination[1], .08), 45),
            ((destination[0], destination[1], .08), 121.5),
            ((destination[0], destination[1], .14), 121.5),
        )
        for phase in range(8):
            joint_states = [pb.getJointState(sim.arm, joint) for joint in sim.arm_joints]
            joint_position = np.asarray([state[0] for state in joint_states], dtype=np.float32)
            joint_velocity = np.asarray([state[1] for state in joint_states], dtype=np.float32)
            gripper = np.asarray([pb.getJointState(sim.arm, joint)[0] for joint in sim.gripper_joints], dtype=np.float32)
            image = sim.image((sim.observation_size, sim.observation_size))
            waypoint, finger_travel_mm = waypoints[phase]
            opening = float(finger_travel_mm / 1000)
            finger_targets = np.clip([opening * (.05 / .1215), opening * (.0715 / .1215)],
                                     [0, 0], [.05, .0715])
            target = np.concatenate((sim._ik(waypoint, orientation), finger_targets)).astype(np.float32)
            arm_lower = np.asarray([bound[0] for bound in LIMITS[:6]])
            arm_upper = np.asarray([bound[1] for bound in LIMITS[:6]])
            target[:6] = np.clip(target[:6], arm_lower + 1e-5, arm_upper - 1e-5)
            proprio = encode_proprioception(joint_position, joint_velocity, gripper)
            tokens = encode_instruction(sim.current_instruction)
            visual = (torch.from_numpy(encode_visual_targets(
                image[None], tokens[None], legacy_image_yaw=visual_yaw_convention == "legacy"))
                if model.visual_feature_dim else None)
            with torch.no_grad():
                prediction = model(torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255,
                                    torch.from_numpy(proprio).unsqueeze(0),
                                    torch.from_numpy(tokens).unsqueeze(0),
                                    torch.tensor([phase], dtype=torch.long),
                                    visual).squeeze(0).numpy()
            teacher_takeover = False
            if action_mode == "cartesian":
                action = denormalize_cartesian_actions(prediction)
                teacher_action = np.asarray([*waypoint, cube_yaw, opening], dtype=np.float32)
                yaw_error = abs((float(action[3] - teacher_action[3]) + np.pi / 4) %
                                (np.pi / 2) - np.pi / 4)
                intervene = (float(np.linalg.norm(action[:3] - teacher_action[:3])) >
                             intervention_position_m or yaw_error > intervention_yaw_rad or
                             abs(float(action[4] - teacher_action[4])) > intervention_gripper_m)
                if intervene:
                    teacher_takeover = True
                    teacher_interventions += 1
            else:
                action = denormalize_actions(prediction)
                action[:6] = np.clip(action[:6], [bound[0] for bound in LIMITS[:6]],
                                     [bound[1] for bound in LIMITS[:6]])
                intervene = (float(np.max(np.abs(action[:6] - target[:6]))) > .20 or
                             float(np.max(np.abs(action[6:] - target[6:]))) > .01)
                if intervene:
                    teacher_takeover = True
                    teacher_interventions += 1
            images.append(image)
            positions.append(joint_position)
            velocities.append(joint_velocity)
            grippers.append(gripper)
            actions.append(target)
            phases.append(phase)
            instructions.append(sim.current_instruction)
            if action_mode == "cartesian":
                if teacher_takeover:
                    sim.move(waypoint, finger_travel_mm=finger_travel_mm, orientation=orientation)
                else:
                    yaw_rotation = pb.getQuaternionFromEuler([0, 0, float(action[3])])
                    orientation = pb.multiplyTransforms([0, 0, 0], yaw_rotation,
                                                        [0, 0, 0], sim.top_down)[1]
                    sim.move(action[:3], finger_travel_mm=float(action[4] * 1000),
                             orientation=orientation)
            else:
                if teacher_takeover:
                    sim.move(waypoint, finger_travel_mm=finger_travel_mm, orientation=orientation)
                else:
                    sim.move_joints(action[:6], finger_targets=action[6:])
        for _ in range(240):
            pb.stepSimulation()
            sim.ticks += 1
        sim.move_joints(sim.home_q)
        final = sim.object_position(name)
        distance = float(np.linalg.norm(final[:2] - PAD_XY[name]))
        object_results.append({"object": name, "success": bool(distance < SUCCESS_TOLERANCE_M and .10 < final[2] < .18),
                              "distance_m": round(distance, 5), "final_position_m": final.round(5).tolist()})
    arrays = {"images": np.stack(images).astype(np.uint8),
              "joint_position": np.stack(positions), "joint_velocity": np.stack(velocities),
              "gripper_position": np.stack(grippers), "actions": np.stack(actions),
              "phases": np.asarray(phases, dtype=np.int64), "ticks": np.zeros(len(phases), dtype=np.int64)}
    episode = {"seed": int(seed), "success": all(row["success"] for row in object_results),
               "objects_placed": sum(row["success"] for row in object_results), "objects_attempted": 3,
               "commands": len(phases), "split": "validation" if seed % 5 == 0 else "train",
               "teacher_interventions": teacher_interventions,
               "file": f"episode-{seed}.npz",
               "teacher": "scripted B601-RS waypoint labels queried on learned-policy states",
               "instruction_per_sample": instructions, "objects": object_results}
    return episode, arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--seed-start", type=int, default=640000)
    parser.add_argument("--intervention-position-m", type=float, default=.025,
                        help="take the scripted waypoint if the Cartesian position error exceeds this value")
    parser.add_argument("--intervention-yaw-rad", type=float, default=.25,
                        help="take the scripted waypoint if the cube-equivalent wrist yaw error exceeds this value")
    parser.add_argument("--intervention-gripper-m", type=float, default=.015,
                        help="take the scripted waypoint if gripper opening error exceeds this value")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.episodes < 1 or (args.output.exists() and any(args.output.iterdir())):
        parser.error("episodes must be positive and output must be empty")
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = TinyB601Policy(spatial_size=checkpoint["spatial_size"],
                           phase_conditioned=checkpoint.get("phase_conditioned", False),
                           action_dim=6 if checkpoint.get("action_mode") == "cartesian" else 8,
                           visual_feature_dim=int(checkpoint.get("visual_feature_dim", 0)))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    image_size = int(checkpoint.get("image_size", 192))
    action_mode = checkpoint.get("action_mode", "joint")
    sim = B601Sim(size=max(384, image_size), observation_size=image_size,
                  camera_view=checkpoint.get("camera_view", "oblique"))
    episodes = []
    try:
        for seed in range(args.seed_start, args.seed_start + args.episodes):
            episode, arrays = collect_episode(
                model, sim, seed, action_mode,
                visual_yaw_convention=checkpoint.get("visual_yaw_convention", "legacy"),
                intervention_position_m=args.intervention_position_m,
                intervention_yaw_rad=args.intervention_yaw_rad,
                intervention_gripper_m=args.intervention_gripper_m)
            np.savez_compressed(args.output / episode["file"], **arrays)
            episodes.append(episode)
            print(json.dumps({"seed": seed, "success": episode["success"],
                              "placed": episode["objects_placed"], "samples": episode["commands"],
                              "teacher_interventions": episode["teacher_interventions"]}), flush=True)
    finally:
        sim.close()
    manifest = {"schema_version": 2, "robot": "Seeed reBot B601-RS URDF in PyBullet",
                "task": "three-cube color sorting on open table",
                "image_size": image_size, "camera_view": checkpoint.get("camera_view", "oblique"),
                "visual_yaw_convention": checkpoint.get("visual_yaw_convention"),
                "source_checkpoint": str(args.checkpoint),
                "action": "scripted teacher labels on policy-visited states; teacher keeps each grasp pose fixed through the pick-and-place sequence and takes control by Cartesian waypoint when predicted position, yaw, or gripper error crosses the intervention threshold",
                "intervention_thresholds": {"position_m": args.intervention_position_m,
                                            "yaw_rad": args.intervention_yaw_rad,
                                            "gripper_m": args.intervention_gripper_m},
                "phase_labels": ["approach", "descend", "close", "lift", "carry", "place", "release", "retreat"],
                "privileged_inputs_used_for_labels_only": ["exact simulator object poses", "simulator pad poses"],
                "episode_split": "seed-based metadata split; training uses only DAgger training episodes and validation stays in the original dataset",
                "episodes": episodes, "episode_count": len(episodes),
                "successful_episodes": sum(ep["success"] for ep in episodes),
                "teacher_interventions": sum(ep["teacher_interventions"] for ep in episodes),
                "total_waypoint_samples": sum(ep["commands"] for ep in episodes)}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"episodes": manifest["episode_count"],
                      "successful_episodes": manifest["successful_episodes"],
                      "samples": manifest["total_waypoint_samples"]}, indent=2))


if __name__ == "__main__":
    main()

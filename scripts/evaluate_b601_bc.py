"""Run the learned B601 waypoint policy and teacher on held-out simulation seeds."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from demo_b601_sort import OBJECTS, OBJECT_TO_PAD, PAD_XY, SLOT_SETS, SUCCESS_TOLERANCE_M, randomize_scene, run_trial  # noqa: E402
from tinyvla.b601_bc import (TinyB601Policy, denormalize_actions, encode_instruction,
                             encode_proprioception, denormalize_cartesian_actions,
                             encode_visual_targets)  # noqa: E402
from tinyvla.sim_b601 import B601Sim, LIMITS, pb  # noqa: E402


def _grasp_contact_audit(sim, object_name):
    """Measure the wrist angle and finger contact locations at the close waypoint."""
    cube = sim.objects[object_name]
    cube_position, cube_orientation = pb.getBasePositionAndOrientation(cube)
    cube_yaw = float(pb.getEulerFromQuaternion(cube_orientation)[2])
    tcp_state = pb.getLinkState(sim.arm, sim.tcp_link, computeForwardKinematics=True)
    inverse_top_down = pb.invertTransform([0, 0, 0], sim.top_down)[1]
    relative_orientation = pb.multiplyTransforms(
        [0, 0, 0], tcp_state[5], [0, 0, 0], inverse_top_down)[1]
    wrist_yaw = float(pb.getEulerFromQuaternion(relative_orientation)[2])
    angle_error = (wrist_yaw - cube_yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4
    inverse_cube = pb.invertTransform(cube_position, cube_orientation)
    contacts = []
    for contact in pb.getContactPoints(bodyA=sim.arm, bodyB=cube):
        link = int(contact[3])
        if link not in sim.gripper_joints:
            continue
        local_position = pb.multiplyTransforms(
            inverse_cube[0], inverse_cube[1], contact[6], [0, 0, 0, 1])[0]
        local_x, local_y, local_z = map(float, local_position)
        corner_proximity = min(abs(local_x), abs(local_y)) / .021
        contacts.append({
            "finger_link": link,
            "cube_local_contact_m": [round(local_x, 5), round(local_y, 5), round(local_z, 5)],
            "corner_proximity": round(float(np.clip(corner_proximity, 0, 1)), 3),
            "normal_force_n": round(float(contact[9]), 3),
        })
    return {
        "tcp_position_m": [round(float(value), 5) for value in tcp_state[4]],
        "cube_center_m": np.round(cube_position, 5).tolist(),
        "face_yaw_error_deg": round(abs(float(np.degrees(angle_error))), 2),
        "contact_count": len(contacts),
        "contacting_fingers": sorted({row["finger_link"] for row in contacts}),
        "mean_corner_proximity": (round(float(np.mean([row["corner_proximity"] for row in contacts])), 3)
                                   if contacts else None),
        "contacts": contacts,
    }


def _overhead_pixel_to_world_xy(sim, image_x, image_y, plane_z=.125):
    """Project a color-mask center onto the table plane using camera calibration."""
    width = height = sim.observation_size
    projection = np.asarray(pb.computeProjectionMatrixFOV(
        42, width / height, .03, 3), dtype=float).reshape(4, 4, order="F")
    view = np.asarray(sim.view, dtype=float).reshape(4, 4, order="F")
    inverse = np.linalg.inv(projection @ view)
    ndc_x, ndc_y = float(image_x), -float(image_y)
    near = inverse @ np.asarray([ndc_x, ndc_y, -1., 1.])
    far = inverse @ np.asarray([ndc_x, ndc_y, 1., 1.])
    near = near[:3] / near[3]
    far = far[:3] / far[3]
    direction = far - near
    if abs(direction[2]) < 1e-8:
        raise ValueError("overhead camera ray is parallel to the grasp plane")
    distance = (plane_z - near[2]) / direction[2]
    point = near + distance * direction
    return point[:2]


def policy_trial(model, sim, seed, action_mode="joint", legacy_image_yaw=True,
                 audit_grasps=False, visual_yaw_correction=False,
                 visual_grasp_correction=False, grasp_height_offset=0.0, frames=None):
    sim.reset(seed)
    _, rng = randomize_scene(sim, seed, SLOT_SETS["v7"], min_target_distance=.075)
    order = [OBJECTS[index] for index in rng.permutation(len(OBJECTS))]
    if frames is not None:
        frames.append(sim.image(sim.size))
    objects = []
    model.eval()
    for name in order:
        pad = OBJECT_TO_PAD[name]
        sim.current_instruction = f"sort the {name} onto the {pad} pad"
        grasp_audit = None
        lift_audit = None
        transport_audits = {}
        visual_yaw = None
        visual_grasp_xy = None
        for phase in range(8):
            joint_states = [pb.getJointState(sim.arm, joint) for joint in sim.arm_joints]
            gripper = [pb.getJointState(sim.arm, joint)[0] for joint in sim.gripper_joints]
            proprio = encode_proprioception([state[0] for state in joint_states],
                                           [state[1] for state in joint_states], gripper)
            image = sim.image((sim.observation_size, sim.observation_size))
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255
            proprio_tensor = torch.from_numpy(proprio).unsqueeze(0)
            instruction_tokens = encode_instruction(sim.current_instruction)
            token_tensor = torch.from_numpy(instruction_tokens).unsqueeze(0)
            visual_features = (torch.from_numpy(encode_visual_targets(
                image[None], instruction_tokens[None],
                legacy_image_yaw=legacy_image_yaw))
                if model.visual_feature_dim else None)
            if (visual_yaw_correction and phase == 0 and visual_features is not None
                    and float(visual_features[0, 6]) > .5):
                visual_yaw = float(np.arctan2(visual_features[0, 2].item(),
                                              visual_features[0, 3].item()) / 4)
            if (visual_grasp_correction and phase == 0 and visual_features is not None
                    and float(visual_features[0, 6]) > .5):
                visual_grasp_xy = _overhead_pixel_to_world_xy(
                    sim, visual_features[0, 0].item(), visual_features[0, 1].item())
                visual_yaw = float(np.arctan2(visual_features[0, 2].item(),
                                              visual_features[0, 3].item()) / 4)
            with torch.no_grad():
                phase_tensor = torch.tensor([phase], dtype=torch.long)
                normalized_action = model(image_tensor, proprio_tensor, token_tensor,
                                          phase_tensor, visual_features).squeeze(0).numpy()
            if action_mode == "cartesian":
                action = denormalize_cartesian_actions(normalized_action)
                if visual_grasp_correction and phase <= 3 and visual_grasp_xy is not None:
                    action[:2] = visual_grasp_xy
                if visual_grasp_correction and phase in (1, 2) and visual_grasp_xy is not None:
                    action[2] += grasp_height_offset
                if visual_yaw is not None:
                    action[3] = visual_yaw
                yaw_rotation = pb.getQuaternionFromEuler([0, 0, float(action[3])])
                orientation = pb.multiplyTransforms([0, 0, 0], yaw_rotation,
                                                    [0, 0, 0], sim.top_down)[1]
                sim.move(action[:3], finger_travel_mm=float(action[4] * 1000),
                         frames=frames, orientation=orientation)
            else:
                action = denormalize_actions(normalized_action)
                arm_target = action[:6]
                for value, (lower, upper) in zip(arm_target, LIMITS[:6]):
                    if not lower <= value <= upper:
                        raise RuntimeError("policy action exceeded a B601 arm joint limit")
                sim.move_joints(arm_target, frames=frames, finger_targets=action[6:])
            if audit_grasps and phase == 2:
                grasp_audit = _grasp_contact_audit(sim, name)
            if audit_grasps and phase == 3:
                lift_audit = _grasp_contact_audit(sim, name)
            if audit_grasps and phase in (4, 5):
                transport_audits[str(phase)] = _grasp_contact_audit(sim, name)
        for _ in range(240):
            pb.stepSimulation()
        sim.move_joints(sim.home_q, frames=frames)
        final = sim.object_position(name)
        distance = float(np.linalg.norm(final[:2] - PAD_XY[name]))
        success = bool(distance < SUCCESS_TOLERANCE_M and .10 < final[2] < .18)
        row = {"object": name, "pad": pad, "success": success,
               "distance_m": round(distance, 5), "final_position_m": final.round(5).tolist()}
        if grasp_audit is not None:
            row["grasp_audit"] = grasp_audit
            row["lift_audit"] = lift_audit
            row["transport_audits"] = transport_audits
        objects.append(row)
    return {"seed": seed, "success": all(row["success"] for row in objects),
            "objects_placed": sum(row["success"] for row in objects), "objects": objects,
            "joint_limits_respected": bool(all(lower - 1e-5 <= value <= upper + 1e-5
                for value, (lower, upper) in zip(sim.q, LIMITS[:6])))}


def pack_evaluation(learned, teacher, args, action_mode):
    return {
        "protocol": f"Matched held-out B601-RS open-table sorting; policy predicts eight {action_mode} waypoints per object and uses a fixed return-home reset; teacher may retry up to three times.",
        "checkpoint": str(args.checkpoint),
        "audit_grasps": args.audit_grasps,
        "visual_yaw_correction": args.visual_yaw_correction,
        "visual_grasp_correction": args.visual_grasp_correction,
        "grasp_height_offset_m": args.grasp_height_offset,
        "seed_range": [args.seed_start, args.seed_start + args.trials - 1],
        "trials": args.trials,
        "completed_trials": len(learned),
        "complete": len(learned) == args.trials,
        "policy": {"successes": sum(row["success"] for row in learned),
                   "objects_placed": sum(row["objects_placed"] for row in learned),
                   "objects_attempted": 3 * len(learned), "records": learned},
        "teacher": {"successes": sum(row["success"] for row in teacher),
                    "objects_placed": sum(sum(obj["success"] for obj in row["objects"]) for row in teacher),
                    "objects_attempted": 3 * len(teacher), "records": teacher},
        "learned_policy_uses_teacher_retries": False,
        "teacher_results_reused": bool(args.teacher_results),
        "simulator_only": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "artifacts" / "b601-bc-v1" / "policy.pt")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=621000)
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "b601-bc-v1" / "heldout-evaluation.json")
    parser.add_argument("--teacher-results", type=Path,
                        help="reuse a matched teacher record file instead of rerunning the teacher")
    parser.add_argument("--audit-grasps", action="store_true",
                        help="record close-phase wrist alignment and finger-to-cube contact locations")
    parser.add_argument("--visual-yaw-correction", action="store_true",
                        help="use the overhead color mask to set face-aligned yaw for each cube")
    parser.add_argument("--visual-grasp-correction", action="store_true",
                        help="use calibrated overhead pixels to set cube-center XY and face yaw on grasp waypoints")
    parser.add_argument("--grasp-height-offset", type=float, default=0.0,
                        help="add a vertical offset in metres to the camera-corrected descend and close waypoints")
    parser.add_argument("--resume", action="store_true",
                        help="continue a partial output file with matching settings")
    args = parser.parse_args()
    if args.trials < 1:
        parser.error("trials must be positive")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    action_mode = checkpoint.get("action_mode", "joint")
    spatial_size = checkpoint.get("spatial_size")
    if spatial_size is None:
        input_width = checkpoint["state_dict"]["head.0.weight"].shape[1]
        phase_conditioned = bool(checkpoint.get("phase_conditioned", False))
        base = 14 + 16 + (8 if phase_conditioned else 0)
        spatial_size = round(((input_width - base) / 48) ** .5)
    else:
        phase_conditioned = bool(checkpoint.get("phase_conditioned", False))
    image_size = args.image_size or int(checkpoint.get("image_size", 96))
    if args.visual_yaw_correction and not checkpoint.get("visual_feature_dim"):
        parser.error("--visual-yaw-correction requires a checkpoint trained with visual target features")
    if args.visual_grasp_correction and not checkpoint.get("visual_feature_dim"):
        parser.error("--visual-grasp-correction requires a checkpoint trained with visual target features")
    if args.visual_grasp_correction and checkpoint.get("camera_view") != "overhead":
        parser.error("--visual-grasp-correction requires an overhead-camera checkpoint")
    if args.grasp_height_offset and not args.visual_grasp_correction:
        parser.error("--grasp-height-offset requires --visual-grasp-correction")
    if args.grasp_height_offset and action_mode != "cartesian":
        parser.error("--grasp-height-offset requires a Cartesian-action checkpoint")
    model = TinyB601Policy(spatial_size=spatial_size, phase_conditioned=phase_conditioned,
                           action_dim=6 if action_mode == "cartesian" else 8,
                           visual_feature_dim=int(checkpoint.get("visual_feature_dim", 0)))
    model.load_state_dict(checkpoint["state_dict"])
    learned, teacher = [], []
    if args.resume:
        if not args.output.is_file():
            parser.error("--resume requires an existing partial output file")
        saved = json.loads(args.output.read_text(encoding="utf-8"))
        expected = {"checkpoint": str(args.checkpoint), "seed_range": [args.seed_start, args.seed_start + args.trials - 1],
                    "trials": args.trials, "audit_grasps": args.audit_grasps,
                    "visual_yaw_correction": args.visual_yaw_correction,
                    "visual_grasp_correction": args.visual_grasp_correction,
                    "grasp_height_offset_m": args.grasp_height_offset}
        if any(saved.get(key) != value for key, value in expected.items()):
            parser.error("partial output settings do not match this run")
        learned = saved["policy"]["records"]
        teacher = saved["teacher"]["records"]
        if len(learned) != len(teacher) or len(learned) > args.trials:
            parser.error("partial output has mismatched or excessive records")
    teacher_by_seed = {}
    if args.teacher_results:
        saved_teacher = json.loads(args.teacher_results.read_text(encoding="utf-8"))
        teacher_by_seed = {row["seed"]: row for row in saved_teacher["teacher"]["records"]}
    teacher_by_seed.update({row["seed"]: row for row in teacher})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sim = B601Sim(size=image_size, observation_size=image_size,
                  camera_view=checkpoint.get("camera_view", "oblique"))
    sim.align_gripper_to_object = True
    try:
        for index in range(len(learned), args.trials):
            seed = args.seed_start + index
            learned_result = policy_trial(
                model, sim, seed, action_mode,
                legacy_image_yaw=checkpoint.get("visual_yaw_convention", "legacy") == "legacy",
                audit_grasps=args.audit_grasps,
                visual_yaw_correction=args.visual_yaw_correction,
                visual_grasp_correction=args.visual_grasp_correction,
                grasp_height_offset=args.grasp_height_offset)
            learned.append(learned_result)
            if seed in teacher_by_seed:
                teacher_result = teacher_by_seed[seed]
            else:
                sim.reset(seed)
                sim.align_gripper_to_object = True
                teacher_result, _ = run_trial(sim, seed, SLOT_SETS["v7"], .14, .20,
                                               return_to_home=True, capture=False)
            teacher.append(teacher_result)
            print(json.dumps({"seed": seed,
                              "policy_success": learned_result["success"],
                              "policy_placed": learned_result["objects_placed"],
                              "teacher_success": teacher_result["success"],
                              "teacher_placed": sum(row["success"] for row in teacher_result["objects"])}), flush=True)
            args.output.write_text(json.dumps(pack_evaluation(learned, teacher, args, action_mode), indent=2) + "\n",
                                   encoding="utf-8")
    finally:
        sim.close()
    result = pack_evaluation(learned, teacher, args, action_mode)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"policy_successes": result["policy"]["successes"],
                      "teacher_successes": result["teacher"]["successes"],
                      "policy_objects_placed": result["policy"]["objects_placed"],
                      "teacher_objects_placed": result["teacher"]["objects_placed"],
                      "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()

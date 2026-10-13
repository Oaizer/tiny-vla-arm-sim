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


def policy_trial(model, sim, seed, action_mode="joint"):
    sim.reset(seed)
    _, rng = randomize_scene(sim, seed, SLOT_SETS["v7"], min_target_distance=.075)
    order = [OBJECTS[index] for index in rng.permutation(len(OBJECTS))]
    objects = []
    model.eval()
    for name in order:
        pad = OBJECT_TO_PAD[name]
        sim.current_instruction = f"sort the {name} onto the {pad} pad"
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
                image[None], instruction_tokens[None])) if model.visual_feature_dim else None)
            with torch.no_grad():
                phase_tensor = torch.tensor([phase], dtype=torch.long)
                normalized_action = model(image_tensor, proprio_tensor, token_tensor,
                                          phase_tensor, visual_features).squeeze(0).numpy()
            if action_mode == "cartesian":
                action = denormalize_cartesian_actions(normalized_action)
                yaw_rotation = pb.getQuaternionFromEuler([0, 0, float(action[3])])
                orientation = pb.multiplyTransforms([0, 0, 0], yaw_rotation,
                                                    [0, 0, 0], sim.top_down)[1]
                sim.move(action[:3], finger_travel_mm=float(action[4] * 1000),
                         orientation=orientation)
            else:
                action = denormalize_actions(normalized_action)
                arm_target = action[:6]
                for value, (lower, upper) in zip(arm_target, LIMITS[:6]):
                    if not lower <= value <= upper:
                        raise RuntimeError("policy action exceeded a B601 arm joint limit")
                sim.move_joints(arm_target, finger_targets=action[6:])
        for _ in range(240):
            pb.stepSimulation()
        sim.move_joints(sim.home_q)
        final = sim.object_position(name)
        distance = float(np.linalg.norm(final[:2] - PAD_XY[name]))
        success = bool(distance < SUCCESS_TOLERANCE_M and .10 < final[2] < .18)
        objects.append({"object": name, "pad": pad, "success": success,
                        "distance_m": round(distance, 5), "final_position_m": final.round(5).tolist()})
    return {"seed": seed, "success": all(row["success"] for row in objects),
            "objects_placed": sum(row["success"] for row in objects), "objects": objects,
            "joint_limits_respected": bool(all(lower - 1e-5 <= value <= upper + 1e-5
                for value, (lower, upper) in zip(sim.q, LIMITS[:6])))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "artifacts" / "b601-bc-v1" / "policy.pt")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=621000)
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "b601-bc-v1" / "heldout-evaluation.json")
    parser.add_argument("--teacher-results", type=Path,
                        help="reuse a matched teacher record file instead of rerunning the teacher")
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
    model = TinyB601Policy(spatial_size=spatial_size, phase_conditioned=phase_conditioned,
                           action_dim=6 if action_mode == "cartesian" else 8,
                           visual_feature_dim=int(checkpoint.get("visual_feature_dim", 0)))
    model.load_state_dict(checkpoint["state_dict"])
    learned, teacher = [], []
    teacher_by_seed = {}
    if args.teacher_results:
        saved_teacher = json.loads(args.teacher_results.read_text(encoding="utf-8"))
        teacher_by_seed = {row["seed"]: row for row in saved_teacher["teacher"]["records"]}
    sim = B601Sim(size=image_size, observation_size=image_size,
                  camera_view=checkpoint.get("camera_view", "oblique"))
    sim.align_gripper_to_object = True
    try:
        for index in range(args.trials):
            seed = args.seed_start + index
            learned_result = policy_trial(model, sim, seed, action_mode)
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
    finally:
        sim.close()
    result = {
        "protocol": f"Matched held-out B601-RS open-table sorting; policy predicts eight {action_mode} waypoints per object and uses a fixed return-home reset; teacher may retry up to three times.",
        "seed_range": [args.seed_start, args.seed_start + args.trials - 1],
        "trials": args.trials,
        "policy": {"successes": sum(row["success"] for row in learned),
                   "objects_placed": sum(row["objects_placed"] for row in learned),
                   "objects_attempted": 3 * args.trials, "records": learned},
        "teacher": {"successes": sum(row["success"] for row in teacher),
                    "objects_placed": sum(sum(obj["success"] for obj in row["objects"]) for row in teacher),
                    "objects_attempted": 3 * args.trials, "records": teacher},
        "learned_policy_uses_teacher_retries": False,
        "teacher_results_reused": bool(args.teacher_results),
        "simulator_only": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"policy_successes": result["policy"]["successes"],
                      "teacher_successes": result["teacher"]["successes"],
                      "policy_objects_placed": result["policy"]["objects_placed"],
                      "teacher_objects_placed": result["teacher"]["objects_placed"],
                      "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()

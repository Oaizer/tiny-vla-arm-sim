"""Render one learned B601 policy trial at portfolio video quality."""
import argparse
from pathlib import Path
import sys

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from demo_b601_sort import OBJECTS, OBJECT_TO_PAD, PAD_XY, SLOT_SETS, SUCCESS_TOLERANCE_M, randomize_scene  # noqa: E402
from tinyvla.b601_bc import (TinyB601Policy, denormalize_cartesian_actions, denormalize_actions,
                             encode_instruction, encode_proprioception, encode_visual_targets)  # noqa: E402
from tinyvla.sim_b601 import B601Sim, LIMITS, pb  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame-stride", type=int, default=24)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    action_mode = checkpoint.get("action_mode", "joint")
    image_size = int(checkpoint.get("image_size", 96))
    model = TinyB601Policy(
        spatial_size=checkpoint.get("spatial_size", 8),
        phase_conditioned=checkpoint.get("phase_conditioned", False),
        action_dim=6 if action_mode == "cartesian" else 8,
        visual_feature_dim=int(checkpoint.get("visual_feature_dim", 0)),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    sim = B601Sim(size=(1280, 720), frame_stride=args.frame_stride,
                  observation_size=image_size, camera_view=checkpoint.get("camera_view", "oblique"))
    frames = []
    records = []
    try:
        sim.reset(args.seed)
        _, rng = randomize_scene(sim, args.seed, SLOT_SETS["v7"], min_target_distance=.075)
        order = [OBJECTS[index] for index in rng.permutation(len(OBJECTS))]
        frames.append(sim.image(sim.size))
        for name in order:
            pad = OBJECT_TO_PAD[name]
            sim.current_instruction = f"sort the {name} onto the {pad} pad"
            for phase in range(8):
                states = [pb.getJointState(sim.arm, joint) for joint in sim.arm_joints]
                gripper = [pb.getJointState(sim.arm, joint)[0] for joint in sim.gripper_joints]
                proprio = encode_proprioception([state[0] for state in states],
                                               [state[1] for state in states], gripper)
                image = sim.image((image_size, image_size))
                instruction = encode_instruction(sim.current_instruction)
                image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255
                state_tensor = torch.from_numpy(proprio).unsqueeze(0)
                token_tensor = torch.from_numpy(instruction).unsqueeze(0)
                visual_tensor = (torch.from_numpy(encode_visual_targets(image[None], instruction[None]))
                                 if model.visual_feature_dim else None)
                with torch.no_grad():
                    prediction = model(image_tensor, state_tensor, token_tensor,
                                       torch.tensor([phase]), visual_tensor).squeeze(0).numpy()
                if action_mode == "cartesian":
                    action = denormalize_cartesian_actions(prediction)
                    wrist = pb.getQuaternionFromEuler([0, 0, float(action[3])])
                    orientation = pb.multiplyTransforms([0, 0, 0], wrist,
                                                        [0, 0, 0], sim.top_down)[1]
                    sim.move(action[:3], finger_travel_mm=float(action[4] * 1000),
                             frames=frames, orientation=orientation)
                else:
                    action = denormalize_actions(prediction)
                    target = action[:6]
                    for value, (low, high) in zip(target, LIMITS[:6]):
                        if not low <= value <= high:
                            raise RuntimeError("policy action exceeded a B601 joint limit")
                    sim.move_joints(target, frames=frames, finger_targets=action[6:])
            for _ in range(240):
                pb.stepSimulation()
                sim.ticks += 1
            sim.move_joints(sim.home_q, frames=frames)
            final = sim.object_position(name)
            distance = float(np.linalg.norm(final[:2] - PAD_XY[name]))
            success = bool(distance < SUCCESS_TOLERANCE_M and .10 < final[2] < .18)
            records.append({"object": name, "success": success, "distance_m": round(distance, 5)})
        success_count = sum(row["success"] for row in records)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with imageio.get_writer(args.output, fps=10, codec="libx264", quality=8) as writer:
            for frame in frames:
                canvas = Image.new("RGB", (frame.shape[1], frame.shape[0] + 48), (20, 24, 27))
                canvas.paste(Image.fromarray(frame), (0, 48))
                draw = ImageDraw.Draw(canvas)
                draw.text((6, 5), "Seeed reBot B601-RS | learned V9 policy | simulation", fill=(245, 245, 238))
                draw.text((6, 22), f"seed {args.seed} | {success_count}/3 cubes placed", fill=(190, 220, 190))
                writer.append_data(np.asarray(canvas))
        print({"seed": args.seed, "objects_placed": success_count, "records": records,
               "frames": len(frames), "video": str(args.output)})
    finally:
        sim.close()


if __name__ == "__main__":
    main()

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

from tinyvla.b601_bc import TinyB601Policy  # noqa: E402
from tinyvla.sim_b601 import B601Sim, pb  # noqa: E402
from evaluate_b601_bc import policy_trial  # noqa: E402


class DualViewFrames(list):
    """Keep the overhead image and add a synchronized oblique camera inset."""
    def __init__(self, sim, side_view):
        super().__init__()
        self.sim = sim
        self.side_view = side_view

    def append(self, overhead):
        canvas = Image.fromarray(overhead)
        side = Image.fromarray(self.sim.image((480, 270), view_override=self.side_view))
        left, top = canvas.width - side.width - 24, canvas.height - side.height - 24
        canvas.paste(side, (left, top))
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((left - 3, top - 3, left + side.width + 2, top + side.height + 2),
                       outline=(245, 245, 238), width=3)
        draw.text((left + 8, top + 8), "OBLIQUE CAMERA", fill=(245, 245, 238),
                  stroke_width=2, stroke_fill=(20, 24, 27))
        super().append(np.asarray(canvas))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame-stride", type=int, default=24)
    parser.add_argument("--label", default="learned policy", help="short label shown in the video")
    parser.add_argument("--dual-view", action="store_true",
                        help="add a synchronized oblique camera inset to the overhead view")
    parser.add_argument("--visual-grasp-correction", action="store_true",
                        help="use calibrated overhead pixels for grasp XY and face yaw")
    parser.add_argument("--grasp-height-offset", type=float, default=0.0,
                        help="add a vertical offset in metres to camera-corrected descend and close waypoints")
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if args.visual_grasp_correction and (not checkpoint.get("visual_feature_dim")
                                         or checkpoint.get("camera_view") != "overhead"):
        parser.error("--visual-grasp-correction requires overhead visual-target features")
    if args.grasp_height_offset and not args.visual_grasp_correction:
        parser.error("--grasp-height-offset requires --visual-grasp-correction")
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
    sim.align_gripper_to_object = True
    frames = []
    try:
        if args.dual_view:
            side_view = pb.computeViewMatrix([.91, -1.05, .92], [.34, 0, .11], [0, 0, 1])
            frames = DualViewFrames(sim, side_view)
        record = policy_trial(
            model, sim, args.seed, action_mode,
            legacy_image_yaw=checkpoint.get("visual_yaw_convention", "legacy") == "legacy",
            visual_grasp_correction=args.visual_grasp_correction,
            grasp_height_offset=args.grasp_height_offset,
            frames=frames)
        records = record["objects"]
        success_count = record["objects_placed"]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with imageio.get_writer(args.output, fps=10, codec="libx264", quality=8) as writer:
            for frame in frames:
                canvas = Image.new("RGB", (frame.shape[1], frame.shape[0] + 48), (20, 24, 27))
                canvas.paste(Image.fromarray(frame), (0, 48))
                draw = ImageDraw.Draw(canvas)
                draw.text((6, 5), f"Seeed reBot B601-RS | {args.label} | simulation", fill=(245, 245, 238))
                draw.text((6, 22), f"seed {args.seed} | {success_count}/3 cubes placed", fill=(190, 220, 190))
                writer.append_data(np.asarray(canvas))
        print({"seed": args.seed, "objects_placed": success_count, "records": records,
               "frames": len(frames), "video": str(args.output)})
    finally:
        sim.close()


if __name__ == "__main__":
    main()

"""Record fresh trials with Seeed's articulated B601-RS URDF dynamics."""
from pathlib import Path
import json
import argparse
import imageio.v2 as imageio
import pybullet as pb
import numpy as np
from PIL import Image, ImageDraw
from tinyvla.sim_b601 import B601Sim

ROOT = Path(__file__).resolve().parents[1]
MEDIA = ROOT / "portfolio" / "media"
OUT = ROOT / "artifacts" / "b601-rs-pilot"
SEEDS = list(range(601000, 601012))
VIDEO_SEEDS = {601000, 601004, 601008}


def place_object(sim, seed):
    rng = np.random.default_rng(seed)
    body = sim.objects["red cube"]
    xyz = [float(rng.uniform(.225, .255)), float(rng.uniform(-.185, -.135)), .125]
    pb.resetBasePositionAndOrientation(body, xyz, pb.getQuaternionFromEuler([0, 0, 0]))
    pb.resetBaseVelocity(body, [0, 0, 0], [0, 0, 0])
    for _ in range(24): pb.stepSimulation()


def write_video(frames, seed, outcome):
    destination = MEDIA / f"b601-rs-seed-{seed}-joint-limited.mp4"
    with imageio.get_writer(destination, fps=20, codec="libx264", quality=8) as writer:
        for frame in frames:
            canvas = Image.new("RGB", (384, 432), (20, 24, 27))
            canvas.paste(Image.fromarray(frame), (0, 44))
            draw = ImageDraw.Draw(canvas)
            draw.text((10, 8), "Seeed reBot B601-RS  |  PyBullet dynamics", fill=(245,245,238))
            draw.text((10, 25), f"Joint-limited teacher trial  |  seed {seed}  |  {outcome}", fill=(190,220,190))
            writer.append_data(np.asarray(canvas))
    return destination.relative_to(ROOT / "portfolio").as_posix()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos-only", action="store_true", help="recreate the three selected clips without rerunning the full result set")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    MEDIA.mkdir(parents=True, exist_ok=True)
    records = []
    sim = B601Sim()
    try:
        for seed in sorted(VIDEO_SEEDS) if args.videos_only else SEEDS:
            sim.reset(seed)
            place_object(sim, seed)
            start = sim.object_position("red cube").round(4).tolist()
            frames = [sim.image(384)] if seed in VIDEO_SEEDS else None
            result = sim.teacher_trial("red cube", "yellow", frames)
            if frames is not None:
                result["video"] = write_video(frames, seed, "SUCCESS" if result["success"] else "FAILED")
            records.append({"seed": seed, "start_m": start, **result})
    finally:
        sim.close()
    success_count = sum(row["success"] for row in records)
    if args.videos_only:
        print(json.dumps({"videos_updated": [r.get("video") for r in records]}, indent=2))
        return
    payload = {"title":"B601-RS joint-limited physics pilot", "model":"Seeed B601-RS URDF", "simulator":"PyBullet", "task":"scripted pick and place of a cube", "seeds":SEEDS, "successes":success_count, "trials":len(records), "records":records}
    (OUT / "results.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (ROOT / "portfolio" / "b601-rs-results.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (OUT / "RESULTS.md").write_text(f"""# B601-RS joint-limited physics pilot

## Question

Can a simple scripted pick-and-place run on the articulated B601-RS model without the Panda arm surrogate?

## Setup

- 12 seeded trials with small randomized starts for one red cube.
- Seeed B601-RS URDF and mesh assets, under CERN-OHL-W-2.0.
- PyBullet rigid-body dynamics, table contact, and physical gripper contact.
- Six arm joints use the URDF limits and torque ratings. Commands use a quintic joint-space profile whose peak reference can reach 2.25 rad/s. Seeed documents a 1.50 rad/s web target limit. The fingers use both URDF prismatic joints, but this pilot does not reproduce the motor-to-width conversion. Self-collision is disabled for performance.
- A simulator-state teacher plans waypoints. This is a mechanics and recording pilot, not a VLA evaluation.

## Result

**{success_count}/{len(records)} trials placed the cube on the target.** See `results.json` for per-seed results and the three video files.

## Limits

The randomized starts cover a narrow area and one cube task. The teacher reads exact object and pad positions from the simulator. No image policy was trained or tested. The results do not show that a learned VLA policy can control this arm.
""", encoding="utf-8")
    print(json.dumps({"successes":success_count,"trials":len(records),"artifacts":str(OUT),"videos":[r.get("video") for r in records if r.get("video")]},indent=2))

if __name__ == "__main__":
    main()

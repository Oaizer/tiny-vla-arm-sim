"""Run a B601-RS simulated pick-and-place task across a low barrier."""
from pathlib import Path
import argparse
import json
import time

import imageio.v2 as imageio
import numpy as np
import pybullet as pb
from PIL import Image, ImageDraw

from tinyvla.sim_b601 import B601Sim, LIMITS, MAX_TARGET_SPEED_RAD_S
from demo_b601_sort import OBJECTS, OBJECT_TO_PAD, PAD_XY, SUCCESS_TOLERANCE_M

ROOT = Path(__file__).resolve().parents[1]
MEDIA = ROOT / "portfolio" / "media"
SEEDS = range(610000, 610012)
SLOTS = np.asarray([(x, y) for x in (.225, .255)
                    for y in (-.20, -.10, 0, .10, .20)], dtype=float)
WALL_X = .30
WALL_TOP = .145
CARRY_HEIGHT = .20
APPROACH_HEIGHT = .14


def add_barrier():
    """Place a low, solid wall between every start and its matching pad."""
    collision = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=[.008, .34, (WALL_TOP - .10) / 2])
    visual = pb.createVisualShape(pb.GEOM_BOX, halfExtents=[.008, .34, (WALL_TOP - .10) / 2],
                                  rgbaColor=[.78, .31, .10, 1])
    return pb.createMultiBody(0, collision, visual, [WALL_X, 0, (.10 + WALL_TOP) / 2])


def make_scene(sim, seed):
    sim.reset(seed)
    wall = add_barrier()
    rng = np.random.default_rng(seed)
    starts = {}
    selected = []
    for idx in rng.permutation(len(SLOTS)):
        xy = SLOTS[idx]
        if all(np.linalg.norm(xy - other) >= .08 for other in selected):
            selected.append(xy)
            if len(selected) == len(OBJECTS):
                break
    for name, xy in zip(OBJECTS, selected):
        yaw = float(rng.uniform(-np.pi, np.pi))
        pb.resetBasePositionAndOrientation(sim.objects[name], [xy[0], xy[1], .125],
                                           pb.getQuaternionFromEuler([0, 0, yaw]))
        starts[name] = {"position_m": [round(float(v), 4) for v in xy], "yaw_rad": round(yaw, 4)}
    for _ in range(24):
        pb.stepSimulation()
    return wall, starts, rng


def write_video(frames, seed, result):
    path = MEDIA / f"b601-rs-obstacle-seed-{seed}.mp4"
    with imageio.get_writer(path, fps=10, codec="libx264", quality=9) as writer:
        for frame in frames:
            height, width = frame.shape[:2]
            canvas = Image.new("RGB", (width, height + 48), (20, 24, 27))
            canvas.paste(Image.fromarray(frame), (0, 48))
            draw = ImageDraw.Draw(canvas)
            draw.text((8, 5), "Seeed reBot B601-RS | PyBullet", fill=(245, 245, 238))
            draw.text((8, 24), f"Barrier transfer | seed {seed} | {result}", fill=(190, 220, 190))
            writer.append_data(np.asarray(canvas))
    return path.relative_to(ROOT / "portfolio").as_posix()


def run_trial(sim, seed, capture=False):
    wall, starts, rng = make_scene(sim, seed)
    frames = [sim.image(sim.size)] if capture else None
    order = [OBJECTS[i] for i in rng.permutation(len(OBJECTS))]
    objects = []
    for name in order:
        target = np.asarray(pb.getBasePositionAndOrientation(sim.pads[OBJECT_TO_PAD[name]])[0])
        attempts = 0
        final = sim.object_position(name)
        distance = float(np.linalg.norm(final[:2] - target[:2]))
        while attempts < 3 and (distance >= SUCCESS_TOLERANCE_M or not .10 < final[2] < .18):
            sim.teacher_trial(name, OBJECT_TO_PAD[name], frames,
                              approach_height=APPROACH_HEIGHT, carry_height=CARRY_HEIGHT,
                              return_to_home=True)
            attempts += 1
            final = sim.object_position(name)
            distance = float(np.linalg.norm(final[:2] - target[:2]))
        objects.append({"object": name, "pad": OBJECT_TO_PAD[name],
                        "success": bool(distance < SUCCESS_TOLERANCE_M and .10 < final[2] < .18),
                        "distance_m": round(distance, 5), "final_position_m": final.round(5).tolist(),
                        "attempts": attempts})
    return {"seed": seed, "starts": starts, "pick_order": order, "objects": objects,
            "success": all(obj["success"] for obj in objects),
            "joint_limits_respected": bool(all(lo - 1e-5 <= value <= hi + 1e-5
                for value, (lo, hi) in zip(sim.q, LIMITS[:6]))),
            "peak_reference_speed_rad_s": round(sim.peak_reference_speed_rad_s, 4)}, frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--videos-only", action="store_true")
    parser.add_argument("--video-seed", type=int, default=SEEDS.start)
    args = parser.parse_args()
    seeds = [args.video_seed] if args.videos_only else list(SEEDS)[:args.trials]
    version = "b601-rs-obstacle-v1"
    out = ROOT / "artifacts" / version
    out.mkdir(parents=True, exist_ok=True)
    MEDIA.mkdir(parents=True, exist_ok=True)
    capture_seeds = set() if args.no_video else {args.video_seed}
    sim = B601Sim(size=384, frame_stride=96)
    records = []
    started = time.time()
    try:
        for seed in seeds:
            row, frames = run_trial(sim, seed, seed in capture_seeds)
            if frames is not None:
                row["video"] = write_video(frames, seed, "SUCCESS" if row["success"] else "FAILED")
            records.append(row)
            print(json.dumps({"seed": seed, "success": row["success"],
                              "placed": sum(obj["success"] for obj in row["objects"])}), flush=True)
    finally:
        sim.close()
    if args.videos_only:
        print(json.dumps({"videos": [row.get("video") for row in records],
                          "outcomes": [row["success"] for row in records]}, indent=2))
        return
    payload = {
        "title": "B601-RS barrier-clear three-object transfer V1",
        "version": version,
        "model": "Seeed B601-RS URDF",
        "simulator": "PyBullet",
        "task": "transfer three colored cubes across a low solid barrier to matching pads",
        "trials": len(records),
        "successes": sum(row["success"] for row in records),
        "objects_placed": sum(obj["success"] for row in records for obj in row["objects"]),
        "objects_attempted": 3 * len(records),
        "success_tolerance_m": SUCCESS_TOLERANCE_M,
        "barrier": {"x_m": WALL_X, "top_m": WALL_TOP, "half_width_m": .008},
        "approach_height_m": APPROACH_HEIGHT,
        "carry_height_m": CARRY_HEIGHT,
        "seed_range": [min(seeds), max(seeds)] if seeds else [],
        "max_target_speed_rad_s": MAX_TARGET_SPEED_RAD_S,
        "records": records,
    }
    (out / "results.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (ROOT / "portfolio" / f"{version}-results.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (out / "RESULTS.md").write_text(
        f"# B601-RS barrier-clear transfer V1\n\n"
        f"**{payload['successes']}/{len(records)}** full trials succeeded; "
        f"**{payload['objects_placed']}/{payload['objects_attempted']}** cubes reached the correct pads.\n\n"
        f"Each trial randomizes three cube starts, yaw, and pick order. Every cube must cross a solid "
        f"{WALL_TOP * 100:.0f} mm barrier and land within {SUCCESS_TOLERANCE_M * 1000:.0f} mm of its matching pad. "
        f"The simulator-state teacher uses a {CARRY_HEIGHT * 1000:.0f} mm carry waypoint and can retry each placement twice.\n\n"
        f"This is scripted PyBullet evidence on the articulated Seeed B601-RS model. It is not a learned-policy result or a real-arm safety test. The sample is small and unpaired.\n",
        encoding="utf-8")
    print(json.dumps({"successes": payload["successes"], "trials": len(records),
                      "objects_placed": payload["objects_placed"],
                      "seconds": round(time.time() - started, 1),
                      "artifact": str(out), "videos": [row.get("video") for row in records if row.get("video")]}, indent=2))


if __name__ == "__main__":
    main()

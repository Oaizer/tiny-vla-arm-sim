"""Run the B601-RS three-object randomized precision-sorting challenge."""
from pathlib import Path
import argparse
import json
import time

import imageio.v2 as imageio
import numpy as np
import pybullet as pb
from PIL import Image, ImageDraw

from tinyvla.sim_b601 import B601Sim, LIMITS, MAX_TARGET_SPEED_RAD_S

ROOT = Path(__file__).resolve().parents[1]
MEDIA = ROOT / "portfolio" / "media"
OBJECT_TO_PAD = {"red cube": "yellow", "blue cube": "purple", "green cube": "cyan"}
OBJECTS = tuple(OBJECT_TO_PAD)
SLOT_SETS = {
    "v2": np.asarray([(x, y) for x in (.225, .285, .345, .405)
                      for y in (-.21, -.14, -.07, 0, .07, .14, .21)], dtype=float),
    "v3": np.asarray([(x, y) for x in (.255, .31, .365)
                      for y in (-.18, -.12, -.06, 0, .06, .12, .18)], dtype=float),
    "v4": np.asarray([(x, y) for x in (.255, .31, .365)
                      for y in (-.18, -.12, -.06, 0, .06, .12, .18)], dtype=float),
    "v5": np.asarray([(x, y) for x in (.225, .255, .285, .31)
                      for y in (-.18, -.12, -.06, 0, .06, .12, .18)], dtype=float),
    "v6": np.asarray([(x, y) for x in (.255, .31, .365)
                      for y in (-.18, -.12, -.06, 0, .06, .12, .18)], dtype=float),
    "v7": np.asarray([(x, y) for x in (.225, .255, .285, .31)
                      for y in (-.18, -.12, -.06, 0, .06, .12, .18)], dtype=float),
    "v8": np.asarray([(x, y) for x in (.255, .285, .31)
                      for y in (-.12, -.06, 0, .06, .12, .18)], dtype=float),
    "v9": np.asarray([(x, y) for x in (.225, .255, .285, .31)
                      for y in (-.18, -.12, -.06, 0, .06, .12, .18)], dtype=float),
}
SEED_RANGES = {"v2": range(602000, 602012), "v3": range(603000, 603012), "v4": range(604000, 604012), "v5": range(605000, 605012), "v6": range(606000, 606012), "v7": range(607000, 607012), "v8": range(608000, 608012), "v9": range(609000, 609012)}
WAYPOINT_HEIGHTS = {"v2": (.22, .20), "v3": (.22, .20), "v4": (.14, .14), "v5": (.14, .20), "v6": (.14, .20), "v7": (.14, .20), "v8": (.14, .20), "v9": (.14, .20)}
SUCCESS_TOLERANCE_M = .035
PAD_XY = {"red cube": np.asarray([.34, -.16]), "blue cube": np.asarray([.34, 0]), "green cube": np.asarray([.34, .16])}


def randomize_scene(sim, seed, slots, min_target_distance=0.0):
    rng = np.random.default_rng(seed)
    assigned = {}
    if min_target_distance <= 0:
        selected = []
        for slot_index in rng.permutation(len(slots)):
            pos = slots[slot_index]
            if all(np.linalg.norm(pos - old) >= .080 for old in selected):
                selected.append(pos)
                if len(selected) == len(OBJECTS):
                    break
        if len(selected) != len(OBJECTS):
            raise RuntimeError("Could not sample three separated object starts")
        assigned = dict(zip(OBJECTS, selected))
    def choose(index):
        if index == len(OBJECTS):
            return True
        name = OBJECTS[index]
        for slot_index in rng.permutation(len(slots)):
            pos = slots[slot_index]
            if np.linalg.norm(pos - PAD_XY[name]) < min_target_distance:
                continue
            if any(np.linalg.norm(pos - old) < .080 for old in assigned.values()):
                continue
            assigned[name] = pos
            if choose(index + 1):
                return True
            del assigned[name]
        return False
    if min_target_distance > 0 and not choose(0):
        raise RuntimeError("Could not sample three separated object starts")
    starts = {}
    for name in OBJECTS:
        xy = assigned[name]
        yaw = float(rng.uniform(-np.pi / 2, np.pi / 2))
        body = sim.objects[name]
        xyz = [float(xy[0]), float(xy[1]), .125]
        pb.resetBasePositionAndOrientation(body, xyz, pb.getQuaternionFromEuler([0, 0, yaw]))
        pb.resetBaseVelocity(body, [0, 0, 0], [0, 0, 0])
        starts[name] = {"position_m": [round(x, 4) for x in xyz], "yaw_rad": round(yaw, 4)}
    for _ in range(24):
        pb.stepSimulation()
    return starts, rng


def write_video(frames, seed, result, version, fps=20):
    destination = MEDIA / f"b601-rs-sort-{version}-seed-{seed}.mp4"
    width = int(frames[0].shape[1])
    height = int(frames[0].shape[0])
    with imageio.get_writer(destination, fps=fps, codec="libx264", quality=8) as writer:
        for frame in frames:
            canvas = Image.new("RGB", (width, height + 48), (20, 24, 27))
            canvas.paste(Image.fromarray(frame), (0, 48))
            draw = ImageDraw.Draw(canvas)
            draw.text((6, 5), "Seeed reBot B601-RS | PyBullet", fill=(245, 245, 238))
            draw.text((6, 22), f"3-color sort | seed {seed} | {result}", fill=(190, 220, 190))
            writer.append_data(np.asarray(canvas))
    return destination.relative_to(ROOT / "portfolio").as_posix()


def run_trial(sim, seed, slots, approach_height, carry_height, return_to_home=False, check_grasp_before_transfer=False, capture=False):
    sim.reset(seed)
    starts, rng = randomize_scene(sim, seed, slots, min_target_distance=.075 if seed >= 607000 else 0.0)
    frames = [sim.image(sim.size)] if capture else None
    # Randomize the pick sequence so that one fixed order cannot hide interference.
    order = [OBJECTS[i] for i in rng.permutation(len(OBJECTS))]
    records = []
    for name in order:
        commands_before = sim.command_count
        attempts = 0
        pickup_failures = 0
        target = np.asarray(pb.getBasePositionAndOrientation(sim.pads[OBJECT_TO_PAD[name]])[0])
        final = sim.object_position(name)
        distance = float(np.linalg.norm(final[:2] - target[:2]))
        while attempts < 3 and (distance >= SUCCESS_TOLERANCE_M or not .10 < final[2] < .18):
            outcome = sim.teacher_trial(name, OBJECT_TO_PAD[name], frames, approach_height=approach_height, carry_height=carry_height, return_to_home=return_to_home, check_grasp_before_transfer=check_grasp_before_transfer)
            pickup_failures += not outcome.get("grasped", True)
            attempts += 1
            final = sim.object_position(name)
            distance = float(np.linalg.norm(final[:2] - target[:2]))
        records.append({"object": name, "pad": OBJECT_TO_PAD[name],
                        "success": bool(distance < SUCCESS_TOLERANCE_M and .10 < final[2] < .18),
                        "distance_m": round(distance, 5), "final_position_m": final.round(5).tolist(),
                        "attempts": attempts, "recovery_attempts": max(0, attempts - 1),
                        "pickup_failures": pickup_failures,
                        "commands": sim.command_count - commands_before})
    return {"seed": seed, "starts": starts, "pick_order": order, "objects": records,
            "success": all(row["success"] for row in records),
            "joint_limits_respected": bool(all(lo - 1e-5 <= value <= hi + 1e-5
                for value, (lo, hi) in zip(sim.q, LIMITS[:6]))),
            "peak_reference_speed_rad_s": round(sim.peak_reference_speed_rad_s, 4),
            "commands": sim.command_count}, frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", choices=("v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9"), default="v3")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--videos-only", action="store_true")
    parser.add_argument("--no-video", action="store_true", help="skip camera capture during evaluation")
    parser.add_argument("--video-seed", type=int, help="seed to use for the selected video")
    args = parser.parse_args()
    version = f"b601-rs-sort-{args.version}"
    out = ROOT / "artifacts" / version
    seeds_all = list(SEED_RANGES[args.version])
    video_seeds = {args.video_seed if args.video_seed is not None else seeds_all[0]}
    seeds = sorted(video_seeds) if args.videos_only else seeds_all[:args.trials]
    slots = SLOT_SETS[args.version]
    approach_height, carry_height = WAYPOINT_HEIGHTS[args.version]
    return_to_home = args.version in {"v6", "v7", "v8", "v9"}
    out.mkdir(parents=True, exist_ok=True)
    MEDIA.mkdir(parents=True, exist_ok=True)
    sim = B601Sim(size=384, frame_stride=96) if args.version in {"v6", "v7", "v8", "v9"} and not args.no_video else B601Sim()
    records = []
    started = time.time()
    try:
        for seed in seeds:
            record, frames = run_trial(sim, seed, slots, approach_height, carry_height, return_to_home, args.version == "v9", seed in video_seeds and not args.no_video)
            if frames is not None:
                record["video"] = write_video(frames, seed, "SUCCESS" if record["success"] else "FAILED", args.version, fps=10 if args.version in {"v6", "v7", "v8", "v9"} else 20)
            records.append(record)
            print(json.dumps({"seed": seed, "success": record["success"],
                              "placed": sum(obj["success"] for obj in record["objects"]),
                              "peak_reference_speed_rad_s": record["peak_reference_speed_rad_s"]}), flush=True)
    finally:
        sim.close()
    if args.videos_only:
        print(json.dumps({"videos": [row.get("video") for row in records],
                          "outcomes": [row["success"] for row in records]}, indent=2))
        return
    successes = sum(row["success"] for row in records)
    payload = {"title": f"B601-RS randomized three-object precision sorting {args.version.upper()}",
               "version": version, "model": "Seeed B601-RS URDF",
               "simulator": "PyBullet", "task": "sort three colored cubes onto color-matched pads",
               "success_tolerance_m": SUCCESS_TOLERANCE_M, "trials": len(records),
               "successes": successes, "objects_placed": sum(sum(obj["success"] for obj in row["objects"]) for row in records),
               "objects_attempted": 3 * len(records), "recovery_attempts": sum(obj["recovery_attempts"] for row in records for obj in row["objects"]),
               "pickup_failures": sum(obj["pickup_failures"] for row in records for obj in row["objects"]),
               "max_placement_error_m": round(max(obj["distance_m"] for row in records for obj in row["objects"]), 5),
               "mean_placement_error_m": round(float(np.mean([obj["distance_m"] for row in records for obj in row["objects"]])), 5),
               "approach_height_m": approach_height, "carry_height_m": carry_height,
               "return_to_home_between_objects": return_to_home,
               "min_start_to_matching_pad_distance_m": .075 if args.version in {"v7", "v8", "v9"} else 0.0,
               "grasp_check_before_transfer": args.version == "v9",
               "max_target_speed_rad_s": MAX_TARGET_SPEED_RAD_S,
               "seed_range": [min(seeds), max(seeds)] if seeds else [], "records": records}
    (out / "results.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (ROOT / "portfolio" / f"{version}-results.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (out / "RESULTS.md").write_text(
        f"# B601-RS randomized three-object precision sorting {args.version.upper()}\n\n"
        f"## Result\n\n**{successes}/{len(records)} complete sorting trials succeeded.** "
        f"{payload['objects_placed']}/{payload['objects_attempted']} objects reached the correct pad within "
        f"{SUCCESS_TOLERANCE_M * 1000:.0f} mm tolerance.\n\n"
        f"## Protocol\n\n- Each seed randomizes three separated cube starts across a {len(slots)}-slot table grid.\n"
        f"- Cube yaw and pick order are randomized. Each cube must reach its color-matched pad.\n"
        f"- One trial succeeds only if all three objects land within {SUCCESS_TOLERANCE_M * 1000:.0f} mm of their pad center.\n"
        f"- The controller checks placement after each pick and makes up to two corrective retries when needed.\n"
        f"- V6-V9 return to the folded joint pose. V7-V9 keep starts at least 75 mm from the matching pad. V9 checks that a cube lifted before starting a transfer.\n"
        f"- {len(records)} fresh seeds are used. The simulator-state teacher reads exact object poses and target pads.\n"
        f"- Joint references use a quintic profile capped at {MAX_TARGET_SPEED_RAD_S:.2f} rad/s. {args.version.upper()} uses {approach_height:.2f} m pre-grasp and {carry_height:.2f} m carry waypoints; this does not model the real motor controller or gripper-width conversion.\n"
        f"- This is scripted simulation evidence, not a learned-policy result or real-arm safety test.\n",
        encoding="utf-8")
    print(json.dumps({"successes": successes, "trials": len(records),
                      "objects_placed": payload["objects_placed"],
                      "objects_attempted": payload["objects_attempted"],
                      "seconds": round(time.time() - started, 1),
                      "artifacts": str(out),
                      "videos": [r.get("video") for r in records if r.get("video")]}, indent=2))


if __name__ == "__main__":
    main()

"""Compare teacher grasp heights and simulator contact geometry on matched scenes."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from demo_b601_sort import (OBJECTS, OBJECT_TO_PAD, PAD_XY, SLOT_SETS,
                            SUCCESS_TOLERANCE_M, randomize_scene)
from evaluate_b601_bc import _grasp_contact_audit
from tinyvla.sim_b601 import B601Sim


def run_object(sim, name, grasp_height):
    metrics = []
    move = sim.move

    def capture_close(*args, **kwargs):
        result = move(*args, **kwargs)
        if sim.current_phase == 2:
            metrics.append(_grasp_contact_audit(sim, name))
        return result

    sim.move = capture_close
    try:
        result = sim.teacher_trial(
            name, OBJECT_TO_PAD[name], approach_height=.14, carry_height=.20,
            return_to_home=True, check_grasp_before_transfer=True,
            grasp_height=grasp_height)
    finally:
        sim.move = move
    final = sim.object_position(name)
    target = np.asarray(pb_position(sim, sim.pads[OBJECT_TO_PAD[name]]))
    distance = float(np.linalg.norm(final[:2] - target[:2]))
    placed = bool(distance < SUCCESS_TOLERANCE_M and .10 < final[2] < .18)
    return {
        "object": name,
        "grasped": bool(result.get("grasped", False)),
        "placed": placed,
        "distance_m": round(distance, 5),
        "final_position_m": final.round(5).tolist(),
        "grasp_contact": metrics[-1] if metrics else None,
    }


def pb_position(sim, body):
    from tinyvla.sim_b601 import pb
    return pb.getBasePositionAndOrientation(body)[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-start", type=int, default=651000)
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--heights", type=float, nargs="+", default=(.08, .09, .10))
    parser.add_argument("--output", type=Path,
                        default=ROOT / "artifacts" / "b601-grasp-height-pilot" / "results.json")
    args = parser.parse_args()
    if args.trials < 1 or any(height <= 0 for height in args.heights):
        parser.error("trials and grasp heights must be positive")

    sim = B601Sim(size=96, observation_size=96, camera_view="overhead")
    records = []
    try:
        for seed in range(args.seed_start, args.seed_start + args.trials):
            for height in args.heights:
                sim.reset(seed)
                sim.align_gripper_to_object = True
                _, rng = randomize_scene(sim, seed, SLOT_SETS["v7"], min_target_distance=.075)
                order = [OBJECTS[index] for index in rng.permutation(len(OBJECTS))]
                objects = [run_object(sim, name, height) for name in order]
                row = {
                    "seed": seed,
                    "grasp_height_m": float(height),
                    "objects": objects,
                    "grasped": sum(obj["grasped"] for obj in objects),
                    "placed": sum(obj["placed"] for obj in objects),
                }
                records.append(row)
                print(json.dumps({"seed": seed, "grasp_height_m": height,
                                  "grasped": row["grasped"], "placed": row["placed"]}), flush=True)
    finally:
        sim.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol": "Matched B601-RS teacher scenes; only pick descend/close TCP height changes. Face yaw alignment is enabled. One attempt per object.",
        "seed_range": [args.seed_start, args.seed_start + args.trials - 1],
        "heights_m": list(args.heights),
        "records": records,
        "simulator_only": True,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "conditions": [{
            "grasp_height_m": height,
            "objects": sum(1 for row in records if row["grasp_height_m"] == height for _ in row["objects"]),
            "grasped": sum(row["grasped"] for row in records if row["grasp_height_m"] == height),
            "placed": sum(row["placed"] for row in records if row["grasp_height_m"] == height),
        } for height in args.heights],
    }, indent=2))


if __name__ == "__main__":
    main()

"""Collect B601-RS scripted-teacher demonstrations for behavior cloning.

Each sample pairs the camera and robot proprioception with the next teacher
joint/gripper waypoint. Simulator object poses are used by the teacher only.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from demo_b601_sort import SLOT_SETS, run_trial  # noqa: E402
from tinyvla.sim_b601 import B601Sim  # noqa: E402

ACTION_LIMITS = np.asarray([
    [-2.8, 2.8], [0, 3.14], [0, 3.14], [-1.57, 1.57],
    [-1.57, 1.57], [-3.14, 3.14], [0, .05], [0, .0715],
], dtype=np.float32)

_WORKER_SIM = None


def _initialize_worker(image_size, camera_view):
    global _WORKER_SIM
    image_size = int(image_size)
    _WORKER_SIM = B601Sim(size=image_size, observation_size=image_size, camera_view=camera_view)


def _collect_to_disk(seed: int, output_dir: str) -> dict:
    episode, arrays = collect_episode(_WORKER_SIM, seed, SLOT_SETS["v7"])
    filename = f"episode-{seed}.npz"
    np.savez_compressed(Path(output_dir) / filename, **arrays)
    episode["file"] = filename
    return episode


def collect_episode(sim: B601Sim, seed: int, slots: np.ndarray) -> tuple[dict, dict]:
    samples: list[dict] = []

    def record(observation, action):
        if observation["phase"] >= 0:
            samples.append({**observation, "action": action})

    sim.command_observer = record
    sim.align_gripper_to_object = True
    result, _ = run_trial(sim, seed, slots, .14, .20, return_to_home=True,
                          check_grasp_before_transfer=False, capture=False)
    sim.command_observer = None
    if not samples:
        raise RuntimeError(f"Teacher produced no samples for seed {seed}")
    episode = {
        "seed": int(seed),
        "success": bool(result["success"]),
        "objects_placed": sum(int(obj["success"]) for obj in result["objects"]),
        "objects_attempted": len(result["objects"]),
        "commands": len(samples),
        "teacher": "B601-RS simulator-state waypoint teacher; labels are commanded joint and finger positions",
        "split": "validation" if seed % 5 == 0 else "train",
        "instruction_per_sample": [sample["instruction"] for sample in samples],
    }
    arrays = {
        "images": np.stack([sample["image"] for sample in samples]).astype(np.uint8),
        "joint_position": np.stack([sample["joint_position"] for sample in samples]),
        "joint_velocity": np.stack([sample["joint_velocity"] for sample in samples]),
        "gripper_position": np.stack([sample["gripper_position"] for sample in samples]),
        "actions": np.clip(np.stack([sample["action"] for sample in samples]),
                            ACTION_LIMITS[:, 0], ACTION_LIMITS[:, 1]),
        "ticks": np.asarray([sample["tick"] for sample in samples], dtype=np.int64),
        "phases": np.asarray([sample["phase"] for sample in samples], dtype=np.int64),
    }
    return episode, arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed-start", type=int, default=620000)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "b601-demos-v2-192")
    parser.add_argument("--workers", type=int, default=1,
                        help="independent PyBullet processes; use 1 for the most predictable CPU load")
    parser.add_argument("--image-size", type=int, default=192,
                        help="square observation image size in pixels (default: 192)")
    parser.add_argument("--camera-view", choices=("oblique", "overhead"), default="oblique")
    args = parser.parse_args()
    if args.episodes < 1 or args.workers < 1 or args.image_size < 64:
        parser.error("episodes and workers must be positive; image-size must be at least 64")
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error(f"output directory is not empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    metadata = []
    seeds = range(args.seed_start, args.seed_start + args.episodes)
    if args.workers == 1:
        sim = B601Sim(size=args.image_size, observation_size=args.image_size, camera_view=args.camera_view)
        try:
            for seed in seeds:
                sim.reset(seed)
                episode, arrays = collect_episode(sim, seed, SLOT_SETS["v7"])
                filename = f"episode-{seed}.npz"
                np.savez_compressed(out / filename, **arrays)
                episode["file"] = filename
                metadata.append(episode)
                print(json.dumps({"seed": seed, "success": episode["success"],
                                  "placed": episode["objects_placed"],
                                  "samples": episode["commands"]}), flush=True)
        finally:
            sim.close()
    else:
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_initialize_worker,
                                 initargs=(args.image_size, args.camera_view)) as pool:
            for episode in pool.map(_collect_to_disk, seeds, [str(out)] * args.episodes, chunksize=1):
                metadata.append(episode)
                print(json.dumps({"seed": episode["seed"], "success": episode["success"],
                                  "placed": episode["objects_placed"],
                                  "samples": episode["commands"]}), flush=True)
        metadata.sort(key=lambda episode: episode["seed"])
    manifest = {
        "schema_version": 1,
        "robot": "Seeed reBot B601-RS URDF in PyBullet",
        "task": "three-cube color sorting on open table",
        "image_size": args.image_size,
        "camera_view": args.camera_view,
        "observation": [f"{args.image_size}x{args.image_size} RGB camera image", "six joint positions", "six joint velocities",
                        "two gripper joint positions", "per-object natural-language instruction"],
        "action": "next six-joint position target plus two gripper joint position targets",
        "action_label_clipping": "labels are clipped to the B601-RS URDF joint limits",
        "grasp_alignment": "scripted teacher rotates the wrist to align finger travel with the cube's randomized yaw",
        "privileged_inputs_used_for_labels_only": ["exact simulator object poses", "simulator pad poses"],
        "episode_split": "validation when seed modulo 5 is zero; otherwise train",
        "episodes": metadata,
        "episode_count": len(metadata),
        "successful_episodes": sum(ep["success"] for ep in metadata),
        "total_waypoint_samples": sum(ep["commands"] for ep in metadata),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(out), "episodes": manifest["episode_count"],
                      "successes": manifest["successful_episodes"],
                      "samples": manifest["total_waypoint_samples"]}, indent=2))


if __name__ == "__main__":
    main()

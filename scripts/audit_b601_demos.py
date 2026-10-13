"""Validate B601 demonstration episode files and leakage-safe splits."""
import argparse
import json
from pathlib import Path

import numpy as np

ACTION_LIMITS = np.asarray([
    [-2.8, 2.8], [0, 3.14], [0, 3.14], [-1.57, 1.57],
    [-1.57, 1.57], [-3.14, 3.14], [0, .05], [0, .0715],
], dtype=np.float32)
ACTION_LIMIT_TOLERANCE = np.asarray([1e-5] * 6 + [3e-5, 3e-5], dtype=np.float32)


def audit(root: Path) -> dict:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    episodes = manifest["episodes"]
    image_size = int(manifest.get("image_size", 96))
    if not episodes:
        raise ValueError("dataset has no episodes")
    seeds = [int(episode["seed"]) for episode in episodes]
    if len(seeds) != len(set(seeds)):
        raise ValueError("episode seeds are duplicated")
    if not any(ep["split"] == "train" for ep in episodes) or not any(ep["split"] == "validation" for ep in episodes):
        raise ValueError("both train and validation episodes are required")
    counts = {"train": 0, "validation": 0}
    successes = 0
    for episode in episodes:
        if episode["split"] != ("validation" if episode["seed"] % 5 == 0 else "train"):
            raise ValueError(f"unexpected split for seed {episode['seed']}")
        with np.load(root / episode["file"], allow_pickle=False) as data:
            required = {"images", "joint_position", "joint_velocity", "gripper_position", "actions", "ticks", "phases"}
            if set(data.files) != required:
                raise ValueError(f"wrong fields in {episode['file']}: {data.files}")
            n = int(data["images"].shape[0])
            expected = {
                "images": (n, image_size, image_size, 3), "joint_position": (n, 6),
                "joint_velocity": (n, 6), "gripper_position": (n, 2),
                "actions": (n, 8), "ticks": (n,), "phases": (n,),
            }
            for name, shape in expected.items():
                if data[name].shape != shape:
                    raise ValueError(f"{episode['file']} {name} has shape {data[name].shape}, expected {shape}")
                if name != "images" and not np.isfinite(data[name]).all():
                    raise ValueError(f"non-finite values in {episode['file']} {name}")
            if n != episode["commands"] or n != len(episode["instruction_per_sample"]):
                raise ValueError(f"sample count mismatch for seed {episode['seed']}")
            actions = data["actions"]
            if np.any(data["phases"] < 0) or np.any(data["phases"] > 7):
                raise ValueError(f"waypoint phase outside 0..7 in {episode['file']}")
            if (np.any(actions < ACTION_LIMITS[:, 0] - ACTION_LIMIT_TOLERANCE)
                    or np.any(actions > ACTION_LIMITS[:, 1] + ACTION_LIMIT_TOLERANCE)):
                raise ValueError(f"action outside B601 joint limits in {episode['file']}")
            counts[episode["split"]] += n
        successes += int(episode["success"])
    if len(episodes) != manifest["episode_count"]:
        raise ValueError("manifest episode_count does not match episode list")
    if successes != manifest["successful_episodes"]:
        raise ValueError("manifest successful_episodes does not match episode list")
    total_samples = sum(counts.values())
    if total_samples != manifest["total_waypoint_samples"]:
        raise ValueError("manifest total_waypoint_samples does not match sample files")
    return {"episodes": len(episodes), "successful_episodes": successes,
            "samples": total_samples, "samples_by_split": counts,
            "validation_seeds": [ep["seed"] for ep in episodes if ep["split"] == "validation"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.dataset), indent=2))


if __name__ == "__main__":
    main()

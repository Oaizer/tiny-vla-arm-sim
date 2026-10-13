"""Add eight-waypoint phase labels and remove scripted home actions from V2 data."""
import argparse
import json
from pathlib import Path

import numpy as np


def prepare(source: Path, destination: Path) -> dict:
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    if destination.exists() and any(destination.iterdir()):
        raise ValueError(f"output directory is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    new_episodes = []
    total = 0
    for episode in manifest["episodes"]:
        with np.load(source / episode["file"], allow_pickle=False) as original:
            count = int(original["actions"].shape[0])
            if count % 9:
                raise ValueError(f"{episode['file']} has {count} commands; expected groups of 9")
            phases = np.tile(np.arange(9, dtype=np.int64), count // 9)
            keep = phases < 8
            arrays = {key: original[key][keep] for key in original.files}
            arrays["phases"] = phases[keep]
            output_file = episode["file"]
            np.savez_compressed(destination / output_file, **arrays)
        updated = dict(episode)
        updated["commands"] = int(keep.sum())
        updated["instruction_per_sample"] = [text for text, retain in zip(
            episode["instruction_per_sample"], keep) if retain]
        updated["file"] = output_file
        new_episodes.append(updated)
        total += updated["commands"]
    result = dict(manifest)
    result["episodes"] = new_episodes
    result["action"] = "next six-joint and two finger-joint waypoint; conditioned on an eight-phase pick-and-place index"
    result["phase_labels"] = ["approach", "descend", "close", "lift", "carry", "place", "release", "retreat"]
    result["excluded_actions"] = "scripted return-home commands; evaluation performs the same deterministic reset outside the policy"
    result["total_waypoint_samples"] = total
    (destination / "manifest.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return {"episodes": len(new_episodes), "samples": total,
            "successful_episodes": result["successful_episodes"], "image_size": result["image_size"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.destination), indent=2))


if __name__ == "__main__":
    main()

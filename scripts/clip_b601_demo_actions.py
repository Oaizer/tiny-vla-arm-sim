"""Clamp recorded waypoint labels to the B601-RS URDF joint limits."""
import argparse
import json
from pathlib import Path

import numpy as np

LIMITS = np.asarray([
    [-2.8, 2.8], [0, 3.14], [0, 3.14], [-1.57, 1.57],
    [-1.57, 1.57], [-3.14, 3.14], [0, .05], [0, .0715],
], dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()
    manifest_path = args.dataset / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    clipped = 0
    for episode in manifest["episodes"]:
        path = args.dataset / episode["file"]
        with np.load(path, allow_pickle=False) as source:
            arrays = {key: source[key] for key in source.files}
        before = arrays["actions"].copy()
        arrays["actions"] = np.clip(before, LIMITS[:, 0], LIMITS[:, 1])
        clipped += int(np.count_nonzero(before != arrays["actions"]))
        temp_path = path.with_suffix(".tmp.npz")
        np.savez_compressed(temp_path, **arrays)
        temp_path.replace(path)
    manifest["action_label_clipping"] = "labels clipped to the B601-RS URDF joint limits"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"episodes": len(manifest["episodes"]), "labels_clipped": clipped,
                      "dataset": str(args.dataset)}, indent=2))


if __name__ == "__main__":
    main()

"""Verify and inventory the frozen world-v3 contact experiment."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts/world-v3/contact"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    expected = {
        "train": (180120, 180359, 240),
        "validation": (181240, 181319, 80),
        "test": (182320, 182399, 80),
        "ranking_v2": (183500, 183539, 40),
    }
    manifest: dict[str, object] = {"experiment": "world-v3-contact", "splits": {}}
    all_seeds: set[int] = set()
    for split, (start, end, count) in expected.items():
        folder = ARTIFACTS / ("ranking_v2" if split == "ranking_v2" else split)
        files = sorted(folder.glob("*.npz"), key=lambda p: int(p.stem))
        seeds = [int(path.stem) for path in files]
        assert len(files) == count, f"{split}: expected {count} files, found {len(files)}"
        assert seeds == list(range(start, end + 1)), f"{split}: seed range mismatch"
        assert not all_seeds.intersection(seeds), f"{split}: seed overlap"
        all_seeds.update(seeds)
        episodes = []
        for path in files:
            with np.load(path, allow_pickle=False) as record:
                actions = record["actions"]
                forces = record["contact_force_n"]
                labels = forces.sum(axis=1) > 0.01
                assert len(actions) + 1 == len(labels), f"{path.name}: transition/observation mismatch"
                assert forces.shape == (len(labels), 4), f"{path.name}: force shape mismatch"
                assert np.array_equal(labels, forces.sum(axis=1) > 0.01), f"{path.name}: label threshold mismatch"
                episodes.append({"seed": int(path.stem), "transitions": int(len(actions)), "sha256": sha256(path)})
        manifest["splits"][split] = {"count": count, "seed_start": start, "seed_end": end, "episodes": episodes}

    training_report = json.loads((ARTIFACTS / "report.json").read_text())
    rank = json.loads((ARTIFACTS / "candidate_rank_v2.json").read_text())
    rank_protocol = json.loads((ARTIFACTS / "candidate_rank_v2_protocol.json").read_text())
    assert len(rank["rows"]) == 40
    assert rank["candidate_branches"] == 1000
    assert len({row["seed"] for row in rank["rows"]}) == 40
    assert all(len(row["actual_contact"]) == 25 for row in rank["rows"])
    assert all(sum(row["actual_contact"]) == row["positive_candidate_count"] for row in rank["rows"])
    assert sha256(ARTIFACTS / "report.json") == rank_protocol["contact_test_report_sha256"]
    for relative, expected_hash in training_report["source_hashes"].items():
        assert sha256(ROOT / relative) == expected_hash, f"training source changed: {relative}"
    for relative, expected_hash in rank_protocol["model_hashes"].items():
        assert sha256(ROOT / relative) == expected_hash, f"ranking input changed: {relative}"
    manifest["ranking"] = {"scenes": 40, "candidate_branches": 1000, "seed_start": 183500, "seed_end": 183539}
    manifest["files"] = {}
    for path in sorted(ARTIFACTS.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            manifest["files"][path.relative_to(ROOT).as_posix()] = sha256(path)
    (ARTIFACTS / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"verified_scenes": len(all_seeds), "ranking_scenes": 40, "candidate_branches": 1000, "manifest": str((ARTIFACTS / 'manifest.json').relative_to(ROOT))}, indent=2))


if __name__ == "__main__":
    main()

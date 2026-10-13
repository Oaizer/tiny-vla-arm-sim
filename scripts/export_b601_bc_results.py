"""Build the compact, seed-level portfolio record from local B601 policy runs."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = [
    ("b601-bc-v1-all", "BC V1", "96 px images; all demonstrations", "b601-bc-v1-all/heldout-evaluation.json"),
    ("b601-bc-v1-success-only", "BC V1 successful-only", "96 px images; complete teacher episodes only", "b601-bc-v1-success-only/heldout-evaluation.json"),
    ("b601-bc-v2-192-all", "BC V2", "192 px images; all demonstrations", "b601-bc-v2-192-all/heldout-evaluation.json"),
    ("b601-bc-v2-192-success-only", "BC V2 successful-only", "192 px images; complete teacher episodes only", "b601-bc-v2-192-success-only/heldout-evaluation.json"),
    ("b601-bc-v3-phase192", "BC V3", "Remove home labels; add a phase input", "b601-bc-v3-phase192/heldout-evaluation.json"),
    ("b601-bc-v4-dagger", "BC V4", "Add 960 teacher labels from policy-visited states", "b601-bc-v4-dagger/heldout-evaluation.json"),
    ("b601-bc-v5-384", "BC V5", "384 px images; 12 by 12 vision grid", "b601-bc-v5-384/heldout-evaluation.json"),
    ("b601-bc-v5b-384-grid8", "BC V5b", "384 px images; smaller 8 by 8 vision grid", "b601-bc-v5b-384-grid8/heldout-evaluation.json"),
    ("b601-bc-v6-cartesian-canonical", "BC V6", "Cartesian waypoint actions; canonical face yaw", "b601-bc-v6-cartesian-canonical/heldout-evaluation.json"),
    ("b601-bc-v7-overhead-cartesian", "BC V7", "Overhead camera; Cartesian waypoints", "b601-bc-v7-overhead-cartesian/heldout-evaluation.json"),
    ("b601-bc-v8-overhead-dagger", "BC V8", "Add 40 episodes of DAgger corrections", "b601-bc-v8-overhead-dagger/heldout-evaluation.json"),
    ("b601-bc-v9-overhead-visual-targets", "BC V9", "Add camera-derived cube and pad coordinates", "b601-bc-v9-overhead-visual-targets/heldout-evaluation.json"),
    ("b601-bc-v10-overhead-visual-dagger", "BC V10", "Combine camera-derived targets with DAgger", "b601-bc-v10-overhead-visual-dagger/heldout-evaluation.json"),
    ("b601-bc-v11-overhead-visual-yaw", "BC V11", "Correct overhead image yaw handedness", "b601-bc-v11-overhead-visual-yaw/heldout-evaluation.json"),
    ("b601-bc-v12-overhead-visual-dagger", "BC V12", "Add corrections from V11-visited states", "b601-bc-v12-overhead-visual-dagger/heldout-evaluation.json"),
    ("b601-bc-v13-overhead-visual-success-only", "BC V13", "Train on successful teacher episodes only", "b601-bc-v13-overhead-visual-success-only/heldout-evaluation.json"),
    ("b601-bc-v14-overhead-visual-grasp-weighted", "BC V14", "Weight grasp-phase waypoint errors 3×", "b601-bc-v14-overhead-visual-grasp-weighted/heldout-evaluation.json"),
]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--output", type=Path, default=ROOT / "portfolio" / "b601-bc-results.json")
    args = parser.parse_args()
    versions = []
    for run_name, label, change, eval_rel in RUNS:
        directory = args.artifacts / run_name
        report_path = directory / "RESULTS.json"
        eval_path = args.artifacts / eval_rel
        if not report_path.is_file() or not eval_path.is_file():
            continue
        report = read_json(report_path)
        dataset_path = Path(report["dataset"])
        if not dataset_path.is_absolute():
            dataset_path = ROOT / dataset_path
        dataset_manifest_path = dataset_path / "manifest.json"
        dataset_manifest = read_json(dataset_manifest_path) if dataset_manifest_path.is_file() else {}
        heldout = read_json(eval_path)
        replications = []
        if run_name in {"b601-bc-v7-overhead-cartesian", "b601-bc-v9-overhead-visual-targets",
                        "b601-bc-v10-overhead-visual-dagger", "b601-bc-v11-overhead-visual-yaw",
                        "b601-bc-v12-overhead-visual-dagger", "b601-bc-v13-overhead-visual-success-only",
                        "b601-bc-v14-overhead-visual-grasp-weighted"}:
            for replication in sorted(directory.glob("heldout-replication-*.json")):
                replications.append(read_json(replication))
        row = {
            "id": run_name,
            "label": label,
            "change": change,
            "action_mode": report.get("action_mode", "joint"),
            "camera_view": dataset_manifest.get("camera_view", "oblique"),
            "image_size": dataset_manifest.get("image_size", 96),
            "visual_target_features": report.get("visual_target_features", False),
            "grasp_phase_weight": report.get("grasp_phase_weight", 1.0),
            "checkpoint_selection_metric": report.get("checkpoint_selection_metric"),
            "validation_weighted_smooth_l1": report.get("validation_weighted_smooth_l1"),
            "training_samples": report["training_samples"],
            "validation_samples": report["validation_samples"],
            "training_episodes": report["training_episodes"],
            "parameters": report["parameters"],
            "checkpoint_bytes": report["checkpoint_bytes"],
            "best_epoch": report["best_epoch"],
            "training_seconds": report["training_seconds"],
            "cpu_latency_ms": report["inference_latency_ms_cpu_batch1"],
            "validation_waypoint_metrics": report["validation_waypoint_metrics"],
            "runs": [heldout, *replications],
        }
        dagger_dirs = {
            "b601-bc-v8-overhead-dagger": "b601-dagger-v8-overhead-40",
            "b601-bc-v10-overhead-visual-dagger": "b601-dagger-v8-overhead-40",
            "b601-bc-v12-overhead-visual-dagger": "b601-dagger-v11-yaw-40",
        }
        if run_name in dagger_dirs:
            dagger_manifest = read_json(args.artifacts / dagger_dirs[run_name] / "manifest.json")
            row["dagger_data"] = {
                "episodes": dagger_manifest["episode_count"],
                "successful_episodes": dagger_manifest["successful_episodes"],
                "samples": dagger_manifest["total_waypoint_samples"],
                "training_samples": sum(ep["commands"] for ep in dagger_manifest["episodes"] if ep["split"] == "train"),
                "validation_samples": sum(ep["commands"] for ep in dagger_manifest["episodes"] if ep["split"] == "validation"),
            }
        versions.append(row)
    payload = {
        "title": "B601-RS learned policy results",
        "updated": "2026-10-13",
        "protocol": "All policies use the Seeed B601-RS URDF in PyBullet. Learned policies receive eight waypoints per object and no retries. Complete-sort success requires all three cubes on their matching pads within 35 mm.",
        "versions": versions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"versions": len(versions), "output": str(args.output),
                      "ids": [row["id"] for row in versions]}, indent=2))


if __name__ == "__main__":
    main()

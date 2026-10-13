"""Build the V18 matched comparison and add it to the B601 portfolio data."""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
OUTPUT = ARTIFACTS / "b601-bc-v18-intervention-dagger"
V17 = ARTIFACTS / "b601-bc-v17-overhead-visual-more-demos-dagger"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def records(folder, block):
    path = folder / ("heldout-evaluation.json" if block == 621000 and
                     folder.name != OUTPUT.name else f"heldout-replication-{block}.json")
    return read_json(path)["policy"]["records"]


def sign_test(wins, losses):
    n = wins + losses
    if not n:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(wins, losses) + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def compare(candidate, reference):
    wins = losses = ties = sort_wins = sort_losses = sort_ties = 0
    by_object = {name: {"candidate": 0, "reference": 0, "trials": 0}
                 for name in ("red cube", "blue cube", "green cube")}
    for left, right in zip(candidate, reference):
        difference = left["objects_placed"] - right["objects_placed"]
        wins += difference > 0
        losses += difference < 0
        ties += difference == 0
        difference = int(left["success"]) - int(right["success"])
        sort_wins += difference > 0
        sort_losses += difference < 0
        sort_ties += difference == 0
        for left_object, right_object in zip(left["objects"], right["objects"]):
            name = left_object["object"]
            by_object[name]["candidate"] += int(left_object["success"])
            by_object[name]["reference"] += int(right_object["success"])
            by_object[name]["trials"] += 1
    return {
        "candidate_cubes": sum(row["objects_placed"] for row in candidate),
        "reference_cubes": sum(row["objects_placed"] for row in reference),
        "candidate_full_sorts": sum(row["success"] for row in candidate),
        "reference_full_sorts": sum(row["success"] for row in reference),
        "wins": wins, "losses": losses, "ties": ties,
        "exact_sign_test_two_sided_p": sign_test(wins, losses),
        "full_sort_wins": sort_wins, "full_sort_losses": sort_losses,
        "full_sort_ties": sort_ties,
        "by_object": by_object,
    }


def main():
    run_files = [OUTPUT / ("heldout-replication-621000.json" if block == 621000 else
                           f"heldout-replication-{block}.json")
                 for block in range(621000, 631000, 1000)]
    runs = [read_json(path) for path in run_files]
    candidate = [row for run in runs for row in run["policy"]["records"]]
    assert len(candidate) == 120
    references = {}
    for label, folder in (
        ("V17", V17),
        ("V16", ARTIFACTS / "b601-bc-v16-overhead-visual-more-demos"),
        ("V11", ARTIFACTS / "b601-bc-v11-overhead-visual-yaw"),
    ):
        references[label] = [row for block in range(621000, 631000, 1000)
                             for row in records(folder, block)]
    for reference in references.values():
        assert [row["seed"] for row in candidate] == [row["seed"] for row in reference]
    comparisons = {label: compare(candidate, reference)
                   for label, reference in references.items()}

    report = read_json(OUTPUT / "RESULTS.json")
    new_datasets = [
        read_json(ARTIFACTS / name / "manifest.json")
        for name in ("b601-dagger-v17-intervention-fixed-pilot-10",
                     "b601-dagger-v17-intervention-fixed-40")
    ]
    new_data = {
        "episodes": sum(item["episode_count"] for item in new_datasets),
        "successful_episodes": sum(item["successful_episodes"] for item in new_datasets),
        "samples": sum(item["total_waypoint_samples"] for item in new_datasets),
        "training_episodes": sum(ep["split"] == "train" for item in new_datasets
                                  for ep in item["episodes"]),
        "training_samples": sum(ep["commands"] for item in new_datasets
                                 for ep in item["episodes"] if ep["split"] == "train"),
        "teacher_interventions": sum(item["teacher_interventions"] for item in new_datasets),
    }
    version = {
        "id": "b601-bc-v18-intervention-dagger",
        "label": "BC V18",
        "change": "Gate large policy errors with teacher takeovers",
        "action_mode": report["action_mode"], "camera_view": "overhead",
        "image_size": report["image_size"] if "image_size" in report else 384,
        "visual_target_features": report["visual_target_features"],
        "grasp_phase_weight": report["grasp_phase_weight"],
        "delivery_phase_weight": report["delivery_phase_weight"],
        "checkpoint_selection_metric": report["checkpoint_selection_metric"],
        "validation_weighted_smooth_l1": None,
        "training_samples": report["training_samples"],
        "validation_samples": report["validation_samples"],
        "training_episodes": report["training_episodes"],
        "parameters": report["parameters"], "checkpoint_bytes": report["checkpoint_bytes"],
        "best_epoch": report["best_epoch"], "training_seconds": report["training_seconds"],
        "cpu_latency_ms": report["inference_latency_ms_cpu_batch1"],
        "validation_waypoint_metrics": report["validation_waypoint_metrics"],
        "dagger_data": {"episodes": new_data["episodes"],
                        "successful_episodes": new_data["successful_episodes"],
                        "samples": new_data["samples"],
                        "training_samples": new_data["training_samples"],
                        "teacher_interventions": new_data["teacher_interventions"]},
        "runs": runs,
        "extra_training_data": [
            {"dataset": r"artifacts\b601-demos-v15-overhead384-extra", "episodes": 100,
             "successful_episodes": 75, "samples": 3448, "training_samples": 2752},
            {"dataset": r"artifacts\b601-dagger-v16-expanded-40", "episodes": 40,
             "successful_episodes": 1, "samples": 960, "training_samples": 768},
            {"dataset": r"artifacts\b601-dagger-v17-intervention-fixed-pilot-10",
             "episodes": 10, "successful_episodes": 2, "samples": 240, "training_samples": 192},
            {"dataset": r"artifacts\b601-dagger-v17-intervention-fixed-40",
             "episodes": 40, "successful_episodes": 7, "samples": 960, "training_samples": 768},
        ],
        "comparison_to_v17": comparisons["V17"],
        "comparison_to_v16": comparisons["V16"],
        "comparison_to_v11": comparisons["V11"],
        "simulator_only": True,
    }
    master_path = ROOT / "portfolio" / "b601-bc-results.json"
    master = read_json(master_path)
    master["versions"] = [item for item in master["versions"]
                          if item["id"] != version["id"]] + [version]
    master_path.write_text(json.dumps(master, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# B601 behavior cloning V18 results",
        "",
        "V18 adds intervention-guided DAgger data. During collection, it records the scripted teacher waypoint at each policy-visited state. If the predicted Cartesian position, cube-equivalent wrist yaw, or gripper opening differs beyond its threshold, the teacher takes over using the same fixed-pose Cartesian path as the established simulator teacher.",
        "",
        "All work used the B601-RS PyBullet model. No physical robot was used.",
        "",
        "## Training",
        "",
        f"- {report['training_episodes']} training episodes and {report['training_samples']:,} samples; the original validation set was kept.",
        f"- {report['parameters']:,} parameters; {report['checkpoint_bytes']:,}-byte checkpoint.",
        f"- Best checkpoint: epoch {report['best_epoch']}; {report['inference_latency_ms_cpu_batch1']:.2f} ms CPU inference per action.",
        f"- New intervention data: {new_data['episodes']} episodes, {new_data['successful_episodes']} complete teacher-guided sorts, {new_data['samples']:,} labels, and {new_data['teacher_interventions']} teacher takeovers.",
        "",
        "## Matched held-out trials",
        "",
        "The policy ran on the same 120 seeds as V11, V16, and V17. Each trial sorts three cubes. The paired test compares each trial's cube count; ties are excluded from the exact two-sided sign test.",
        "",
        "| Comparison | V18 cubes | Reference cubes | Full sorts | Wins / losses / ties | Exact p |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label in ("V17", "V16", "V11"):
        item = comparisons[label]
        lines.append(f"| V18 vs {label} | {item['candidate_cubes']}/360 | {item['reference_cubes']}/360 | {item['candidate_full_sorts']} vs {item['reference_full_sorts']} | {item['wins']} / {item['losses']} / {item['ties']} | {item['exact_sign_test_two_sided_p']:.3f} |")
    lines += [
        "",
        "Per-object placement counts against V17 show where the change helped and where it regressed:",
        "",
        "| Cube | V18 | V17 |",
        "|---|---:|---:|",
    ]
    for name, data in comparisons["V17"]["by_object"].items():
        lines.append(f"| {name} | {data['candidate']}/{data['trials']} | {data['reference']}/{data['trials']} |")
    lines += [
        "",
        f"V18 has the highest observed count so far: {comparisons['V17']['candidate_cubes']}/360 cubes and {comparisons['V17']['candidate_full_sorts']}/120 full sorts. The paired gain over V17 is not established (p = {comparisons['V17']['exact_sign_test_two_sided_p']:.3f}). The comparison supports further testing, not a confirmed improvement.",
        "",
        "## Reproduction",
        "",
        "Run `python scripts/report_b601_v18.py` after training and evaluating the same ten seed blocks. The ten per-block policy and teacher records are stored beside the checkpoint.",
        "",
    ]
    (OUTPUT / "COMPARISON.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"comparison_to_v17": comparisons["V17"],
                      "comparison_to_v16": comparisons["V16"],
                      "comparison_to_v11": comparisons["V11"],
                      "new_data": new_data}, indent=2))


if __name__ == "__main__":
    main()

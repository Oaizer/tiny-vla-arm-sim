"""Report the fresh-seed comparison for B601 V19 and update portfolio records."""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
V17 = ARTIFACTS / "b601-bc-v17-overhead-visual-more-demos-dagger"
V18 = ARTIFACTS / "b601-bc-v18-intervention-dagger"
V19 = ARTIFACTS / "b601-bc-v19-blue-weight"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def exact_sign_p(wins, losses):
    count = wins + losses
    if not count:
        return 1.0
    tail = sum(math.comb(count, index)
               for index in range(min(wins, losses) + 1)) / 2 ** count
    return min(1.0, 2 * tail)


def compare(candidate, reference):
    wins = losses = ties = 0
    sort_wins = sort_losses = sort_ties = 0
    by_color = {name: {"candidate": 0, "reference": 0, "trials": 0}
                for name in ("red cube", "blue cube", "green cube")}
    for left, right in zip(candidate, reference):
        assert left["seed"] == right["seed"]
        difference = left["objects_placed"] - right["objects_placed"]
        wins += difference > 0
        losses += difference < 0
        ties += difference == 0
        difference = int(left["success"]) - int(right["success"])
        sort_wins += difference > 0
        sort_losses += difference < 0
        sort_ties += difference == 0
        right_objects = {row["object"]: row for row in right["objects"]}
        for left_object in left["objects"]:
            name = left_object["object"]
            right_object = right_objects[name]
            by_color[name]["candidate"] += int(left_object["success"])
            by_color[name]["reference"] += int(right_object["success"])
            by_color[name]["trials"] += 1
    return {
        "candidate_cubes": sum(row["objects_placed"] for row in candidate),
        "reference_cubes": sum(row["objects_placed"] for row in reference),
        "candidate_full_sorts": sum(row["success"] for row in candidate),
        "reference_full_sorts": sum(row["success"] for row in reference),
        "wins": wins, "losses": losses, "ties": ties,
        "exact_sign_test_two_sided_p": exact_sign_p(wins, losses),
        "full_sort_wins": sort_wins, "full_sort_losses": sort_losses,
        "full_sort_ties": sort_ties, "by_color": by_color,
    }


def fresh_runs(folder):
    return [load(folder / f"fresh-v19-test-{block}.json")
            for block in range(631000, 641000, 1000)]


def append_fresh_runs(version, runs):
    existing = {tuple(run["seed_range"]) for run in version["runs"]}
    for run in runs:
        key = tuple(run["seed_range"])
        if key not in existing:
            version["runs"].append(run)
            existing.add(key)


def main():
    runs17, runs18, runs19 = fresh_runs(V17), fresh_runs(V18), fresh_runs(V19)
    records17 = [row for run in runs17 for row in run["policy"]["records"]]
    records18 = [row for run in runs18 for row in run["policy"]["records"]]
    records19 = [row for run in runs19 for row in run["policy"]["records"]]
    assert len(records17) == len(records18) == len(records19) == 120
    assert [row["seed"] for row in records17] == [row["seed"] for row in records18]
    assert [row["seed"] for row in records18] == [row["seed"] for row in records19]
    comparison18 = compare(records19, records18)
    comparison17 = compare(records19, records17)

    master_path = ROOT / "portfolio" / "b601-bc-results.json"
    master = load(master_path)
    versions = {version["id"]: version for version in master["versions"]}
    v17_version = versions["b601-bc-v17-overhead-visual-more-demos-dagger"]
    v18_version = versions["b601-bc-v18-intervention-dagger"]
    v17_version["fresh_extension_comparison_to_v18"] = compare(records17, records18)
    v18_version["fresh_extension_comparison_to_v17"] = compare(records18, records17)
    v17_version["fresh_extension_runs"] = runs17
    v18_version["fresh_extension_runs"] = runs18

    report = load(V19 / "RESULTS.json")
    new_data = {
        "episodes": 50, "successful_episodes": 9, "samples": 1200,
        "training_samples": 960, "training_episodes": 40,
        "teacher_interventions": 555,
    }
    version19 = {
        "id": "b601-bc-v19-blue-weight",
        "label": "BC V19",
        "change": "Weight blue-cube instruction samples 1.5 times",
        "action_mode": report["action_mode"], "camera_view": "overhead",
        "image_size": 384,
        "visual_target_features": report["visual_target_features"],
        "grasp_phase_weight": report["grasp_phase_weight"],
        "delivery_phase_weight": report["delivery_phase_weight"],
        "blue_cube_weight": report["blue_cube_weight"],
        "checkpoint_selection_metric": report["checkpoint_selection_metric"],
        "validation_weighted_smooth_l1": None,
        "training_samples": report["training_samples"],
        "validation_samples": report["validation_samples"],
        "training_episodes": report["training_episodes"],
        "parameters": report["parameters"],
        "checkpoint_bytes": report["checkpoint_bytes"],
        "best_epoch": report["best_epoch"],
        "training_seconds": report["training_seconds"],
        "cpu_latency_ms": report["inference_latency_ms_cpu_batch1"],
        "validation_waypoint_metrics": report["validation_waypoint_metrics"],
        "dagger_data": new_data,
        "runs": runs19,
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
        "comparison_to_v18": comparison18,
        "comparison_to_v17": comparison17,
        "simulator_only": True,
    }
    master["versions"] = [version for version in master["versions"]
                           if version["id"] != version19["id"]] + [version19]
    master_path.write_text(json.dumps(master, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# B601 behavior cloning V19 results",
        "",
        "V19 repeats V18's training data, model, optimizer, and schedule. It multiplies the loss for blue-cube instructions by 1.5. The checkpoint was selected by the same unweighted validation score as V18.",
        "",
        "All tests use the B601-RS PyBullet model. No physical arm was used.",
        "",
        "## Training",
        "",
        f"- {report['training_episodes']} training episodes and {report['training_samples']:,} samples.",
        f"- {report['parameters']:,} parameters; {report['checkpoint_bytes']:,}-byte checkpoint.",
        f"- Best checkpoint: epoch {report['best_epoch']}; {report['inference_latency_ms_cpu_batch1']:.2f} ms CPU inference per action.",
        "",
        "## Fresh matched test",
        "",
        "V17, V18, and V19 ran on the same 120 fresh scenes (seeds 631000–640011). The paired sign test compares each trial's cube-placement count and excludes ties.",
        "",
        "| Model | Cubes | Full sorts |",
        "|---|---:|---:|",
        f"| V19 | {comparison18['candidate_cubes']}/360 | {comparison18['candidate_full_sorts']}/120 |",
        f"| V18 | {comparison18['reference_cubes']}/360 | {comparison18['reference_full_sorts']}/120 |",
        f"| V17 | {comparison17['reference_cubes']}/360 | {comparison17['reference_full_sorts']}/120 |",
        "",
        "| Pair | V19 wins / losses / ties | Exact two-sided p |",
        "|---|---:|---:|",
        f"| V19 vs V18 | {comparison18['wins']} / {comparison18['losses']} / {comparison18['ties']} | {comparison18['exact_sign_test_two_sided_p']:.3f} |",
        f"| V19 vs V17 | {comparison17['wins']} / {comparison17['losses']} / {comparison17['ties']} | {comparison17['exact_sign_test_two_sided_p']:.3f} |",
        "",
        "Per-color placement counts against V18:",
        "",
        "| Cube | V19 | V18 |",
        "|---|---:|---:|",
    ]
    for name, count in comparison18["by_color"].items():
        lines.append(f"| {name} | {count['candidate']}/{count['trials']} | {count['reference']}/{count['trials']} |")
    lines += [
        "",
        f"V19 placed {comparison18['candidate_cubes']}/360 cubes, compared with {comparison18['reference_cubes']}/360 for V18 (p = {comparison18['exact_sign_test_two_sided_p']:.3f}). This difference is not conclusive. Against V17, V19 placed {comparison17['candidate_cubes']}/360 versus {comparison17['reference_cubes']}/360 (p = {comparison17['exact_sign_test_two_sided_p']:.3f}). The blue weighting improved the fresh-seed aggregate over V17, but further replication is needed to confirm the gain over V18.",
        "",
        "The full seed-level records are stored beside the checkpoint. Run `python scripts/report_b601_v19.py` to rebuild this report and portfolio data after the same three-way test.",
        "",
    ]
    (V19 / "COMPARISON.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"v19_vs_v18": comparison18, "v19_vs_v17": comparison17}, indent=2))


if __name__ == "__main__":
    main()

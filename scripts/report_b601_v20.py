"""Summarise the matched V20 grasp-height policy evaluation."""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "artifacts" / "b601-v20-evaluation"


def exact_sign_p(wins, losses):
    count = wins + losses
    if not count:
        return 1.0
    tail = sum(math.comb(count, index)
               for index in range(min(wins, losses) + 1)) / 2 ** count
    return min(1.0, 2 * tail)


def summarize_contacts(records):
    objects = [obj for record in records for obj in record["objects"]]
    audits = [obj["grasp_audit"] for obj in objects]
    contacts = [contact for audit in audits for contact in audit["contacts"]]
    corner_values = [audit["mean_corner_proximity"] for audit in audits
                     if audit["mean_corner_proximity"] is not None]
    by_outcome = {}
    for success in (True, False):
        values = [obj["grasp_audit"]["mean_corner_proximity"] for obj in objects
                  if obj["success"] == success
                  and obj["grasp_audit"]["mean_corner_proximity"] is not None]
        by_outcome["success" if success else "failure"] = {
            "objects_with_contact": len(values),
            "mean_corner_proximity": sum(values) / len(values) if values else None,
        }
    return {
        "objects_attempted": len(objects),
        "objects_with_no_finger_contact": sum(audit["contact_count"] == 0 for audit in audits),
        "objects_with_both_fingers_contacting": sum(
            len(audit["contacting_fingers"]) >= 2 for audit in audits),
        "mean_face_yaw_error_deg": sum(audit["face_yaw_error_deg"] for audit in audits) / len(audits),
        "mean_corner_proximity": sum(corner_values) / len(corner_values) if corner_values else None,
        "objects_with_corner_proximity_above_0_65": sum(value > .65 for value in corner_values),
        "finger_contacts": len(contacts),
        "contacts_within_5mm_of_top_or_bottom_rim": sum(
            abs(contact["cube_local_contact_m"][2]) >= .015 for contact in contacts),
        "contact_geometry_by_outcome": by_outcome,
    }


def main():
    v19 = json.loads((EVAL / "v19-661000-120.json").read_text(encoding="utf-8"))
    v20 = json.loads((EVAL / "v20-661000-120.json").read_text(encoding="utf-8"))
    old = v19["policy"]["records"]
    new = v20["policy"]["records"]
    assert len(old) == len(new) == 120
    wins = losses = ties = sort_wins = sort_losses = sort_ties = 0
    colors = {name: {"v19": 0, "v20": 0} for name in ("red cube", "blue cube", "green cube")}
    seeds = []
    for left, right in zip(old, new):
        assert left["seed"] == right["seed"]
        difference = right["objects_placed"] - left["objects_placed"]
        wins += difference > 0
        losses += difference < 0
        ties += difference == 0
        difference = int(right["success"]) - int(left["success"])
        sort_wins += difference > 0
        sort_losses += difference < 0
        sort_ties += difference == 0
        old_objects = {obj["object"]: obj for obj in left["objects"]}
        for obj in right["objects"]:
            colors[obj["object"]]["v20"] += int(obj["success"])
            colors[obj["object"]]["v19"] += int(old_objects[obj["object"]]["success"])
        seeds.append({"seed": left["seed"], "v19_cubes": left["objects_placed"],
                      "v20_cubes": right["objects_placed"],
                      "v19_full_sort": left["success"], "v20_full_sort": right["success"]})
    summary = {
        "title": "B601 V20 high grasp waypoint evaluation",
        "protocol": "120 matched held-out B601-RS PyBullet scenes; seeds 661000–661119; one attempt per object; same saved teacher records; contact audit enabled.",
        "simulator_only": True,
        "checkpoint": {
            "version": "V20",
            "training_episodes": 200,
            "training_samples": 5440,
            "dagger_episodes": 50,
            "parameters": 427798,
            "checkpoint_bytes": 1716137,
            "best_epoch": 30,
            "validation_weighted_smooth_l1": 0.004986605165820373,
            "inference_latency_ms_cpu_batch1": 4.976,
            "pick_height_m": 0.10,
        },
        "v19": {
            "full_sorts": v19["policy"]["successes"], "trials": 120,
            "cubes_placed": v19["policy"]["objects_placed"], "cubes_attempted": 360,
            "contact_audit": summarize_contacts(old),
        },
        "v20": {
            "full_sorts": v20["policy"]["successes"], "trials": 120,
            "cubes_placed": v20["policy"]["objects_placed"], "cubes_attempted": 360,
            "contact_audit": summarize_contacts(new),
        },
        "paired_placements_v20_vs_v19": {
            "wins": wins, "losses": losses, "ties": ties,
            "exact_sign_test_two_sided_p": exact_sign_p(wins, losses),
            "difference_cubes": v20["policy"]["objects_placed"] - v19["policy"]["objects_placed"],
        },
        "paired_full_sorts_v20_vs_v19": {
            "wins": sort_wins, "losses": sort_losses, "ties": sort_ties,
            "exact_sign_test_two_sided_p": exact_sign_p(sort_wins, sort_losses),
        },
        "placements_by_color": colors,
        "teacher_reference": {
            "full_sorts": v19["teacher"]["successes"],
            "cubes_placed": v19["teacher"]["objects_placed"],
            "cubes_attempted": v19["teacher"]["objects_attempted"],
        },
        "paired_seed_results": seeds,
    }
    destination = ROOT / "portfolio" / "b601-v20-evaluation.json"
    destination.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(destination),
                      "v19_cubes": summary["v19"]["cubes_placed"],
                      "v20_cubes": summary["v20"]["cubes_placed"],
                      "paired": summary["paired_placements_v20_vs_v19"]}, indent=2))


if __name__ == "__main__":
    main()

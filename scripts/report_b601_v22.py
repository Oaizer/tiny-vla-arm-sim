"""Summarise the matched camera-corrected B601 V22 evaluation."""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "artifacts" / "b601-v22-evaluation"


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
    corners = [audit["mean_corner_proximity"] for audit in audits
               if audit["mean_corner_proximity"] is not None]
    return {
        "objects_attempted": len(objects),
        "objects_with_no_finger_contact": sum(audit["contact_count"] == 0 for audit in audits),
        "objects_with_both_fingers_contacting": sum(
            len(audit["contacting_fingers"]) >= 2 for audit in audits),
        "mean_face_yaw_error_deg": sum(audit["face_yaw_error_deg"] for audit in audits) / len(audits),
        "mean_corner_proximity": sum(corners) / len(corners) if corners else None,
        "objects_with_corner_proximity_above_0_65": sum(value > .65 for value in corners),
        "finger_contacts": len(contacts),
        "contacts_within_5mm_of_top_or_bottom_rim": sum(
            abs(contact["cube_local_contact_m"][2]) >= .015 for contact in contacts),
    }


def main():
    baseline = json.loads((EVAL / "v19-673000-120.json").read_text(encoding="utf-8"))
    candidate = json.loads((EVAL / "v22-673000-120.json").read_text(encoding="utf-8"))
    old = baseline["policy"]["records"]
    new = candidate["policy"]["records"]
    assert len(old) == len(new) == 120
    wins = losses = ties = 0
    sort_wins = sort_losses = sort_ties = 0
    colors = {name: {"v19": 0, "v22": 0} for name in ("red cube", "blue cube", "green cube")}
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
            colors[obj["object"]]["v22"] += int(obj["success"])
            colors[obj["object"]]["v19"] += int(old_objects[obj["object"]]["success"])
        seeds.append({"seed": left["seed"], "v19_cubes": left["objects_placed"],
                      "v22_cubes": right["objects_placed"],
                      "v19_full_sort": left["success"], "v22_full_sort": right["success"]})
    summary = {
        "title": "B601 V22 camera-corrected grasp evaluation",
        "protocol": "120 fresh matched B601-RS PyBullet scenes; seeds 673000–673119; same V19 checkpoint and teacher records; calibrated overhead color-mask correction on grasp waypoints; contact audit enabled.",
        "simulator_only": True,
        "candidate": {
            "version": "V22",
            "base_checkpoint": "V19",
            "change": "Replace model grasp-waypoint XY and yaw with overhead color-mask estimates projected through the calibrated camera; retain model TCP height, gripper width, carry, place, and release outputs.",
            "training": False,
            "parameters": 427798,
        },
        "v19": {
            "full_sorts": baseline["policy"]["successes"], "trials": 120,
            "cubes_placed": baseline["policy"]["objects_placed"], "cubes_attempted": 360,
            "contact_audit": summarize_contacts(old),
        },
        "v22": {
            "full_sorts": candidate["policy"]["successes"], "trials": 120,
            "cubes_placed": candidate["policy"]["objects_placed"], "cubes_attempted": 360,
            "contact_audit": summarize_contacts(new),
        },
        "paired_placements_v22_vs_v19": {
            "wins": wins, "losses": losses, "ties": ties,
            "exact_sign_test_two_sided_p": exact_sign_p(wins, losses),
            "difference_cubes": candidate["policy"]["objects_placed"] - baseline["policy"]["objects_placed"],
        },
        "paired_full_sorts_v22_vs_v19": {
            "wins": sort_wins, "losses": sort_losses, "ties": sort_ties,
            "exact_sign_test_two_sided_p": exact_sign_p(sort_wins, sort_losses),
        },
        "placements_by_color": colors,
        "teacher_reference": {
            "full_sorts": baseline["teacher"]["successes"],
            "cubes_placed": baseline["teacher"]["objects_placed"],
            "cubes_attempted": baseline["teacher"]["objects_attempted"],
        },
        "paired_seed_results": seeds,
    }
    destination = ROOT / "portfolio" / "b601-v22-evaluation.json"
    destination.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(destination),
                      "v19_cubes": summary["v19"]["cubes_placed"],
                      "v22_cubes": summary["v22"]["cubes_placed"],
                      "paired": summary["paired_placements_v22_vs_v19"]}, indent=2))


if __name__ == "__main__":
    main()

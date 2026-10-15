"""Summarise the B601 V23 grasp-height pilot and held-out test."""
import json
from pathlib import Path

from report_b601_v22 import exact_sign_p

ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "artifacts" / "b601-v23-height-pilot"
TEST = ROOT / "artifacts" / "b601-v23-height-test"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def contact_summary(records):
    objects = [obj for record in records for obj in record["objects"]]
    audits = [obj["grasp_audit"] for obj in objects]
    contacts = [contact for audit in audits for contact in audit["contacts"]]
    heights = [contact["cube_local_contact_m"][2] for contact in contacts]
    corners = [audit["mean_corner_proximity"] for audit in audits
               if audit["mean_corner_proximity"] is not None]
    return {
        "objects_attempted": len(objects),
        "objects_with_no_finger_contact": sum(audit["contact_count"] == 0 for audit in audits),
        "objects_with_both_fingers_contacting": sum(len(audit["contacting_fingers"]) >= 2 for audit in audits),
        "mean_face_yaw_error_deg": sum(audit["face_yaw_error_deg"] for audit in audits) / len(audits),
        "mean_corner_proximity": sum(corners) / len(corners) if corners else None,
        "mean_absolute_contact_height_m": sum(abs(value) for value in heights) / len(heights) if heights else None,
        "median_absolute_contact_height_m": sorted(abs(value) for value in heights)[len(heights) // 2] if heights else None,
        "contacts_within_10mm_of_vertical_face_center": sum(abs(value) <= .01 for value in heights),
        "contacts_within_5mm_of_top_or_bottom_rim": sum(abs(value) >= .015 for value in heights),
        "finger_contacts": len(contacts),
        "trials_with_joint_limit_violation": sum(not record["joint_limits_respected"] for record in records),
    }


def summarize_run(run):
    records = run["policy"]["records"]
    return {
        "trials": len(records),
        "full_sorts": sum(record["success"] for record in records),
        "cubes_placed": sum(record["objects_placed"] for record in records),
        "cubes_attempted": 3 * len(records),
        "contact_audit": contact_summary(records),
    }


def compare(reference, candidate):
    old = reference["policy"]["records"]
    new = candidate["policy"]["records"]
    assert len(old) == len(new) and old
    placement_wins = placement_losses = sort_wins = sort_losses = ties = sort_ties = 0
    seeds = []
    for left, right in zip(old, new):
        assert left["seed"] == right["seed"]
        delta = right["objects_placed"] - left["objects_placed"]
        placement_wins += delta > 0
        placement_losses += delta < 0
        ties += delta == 0
        delta = int(right["success"]) - int(left["success"])
        sort_wins += delta > 0
        sort_losses += delta < 0
        sort_ties += delta == 0
        seeds.append({"seed": left["seed"], "v22_cubes": left["objects_placed"],
                      "v23_cubes": right["objects_placed"],
                      "v22_full_sort": left["success"], "v23_full_sort": right["success"]})
    return {
        "seed_range": [old[0]["seed"], old[-1]["seed"]],
        "v22": summarize_run(reference),
        "v23": summarize_run(candidate),
        "paired_placements_v23_vs_v22": {
            "wins": placement_wins, "losses": placement_losses, "ties": ties,
            "exact_sign_test_two_sided_p": exact_sign_p(placement_wins, placement_losses),
            "difference_cubes": candidate["policy"]["objects_placed"] - reference["policy"]["objects_placed"]},
        "paired_full_sorts_v23_vs_v22": {
            "wins": sort_wins, "losses": sort_losses, "ties": sort_ties,
            "exact_sign_test_two_sided_p": exact_sign_p(sort_wins, sort_losses)},
        "paired_seed_results": seeds,
    }


def main():
    base = load(PILOT / "v22-base-675000-12.json")
    pilot_offsets = {}
    for label, filename, offset in (
            ("minus_10mm", "v22-height-minus-10mm-675000-12.json", -.01),
            ("plus_10mm", "v22-height-plus-10mm-675000-12.json", .01),
            ("plus_20mm", "v22-height-plus-20mm-675000-12.json", .02)):
        candidate = load(PILOT / filename)
        pilot_offsets[label] = {"height_offset_m": offset, **compare(base, candidate)}
    result = {
        "title": "B601 V23 grasp-height correction",
        "simulator_only": True,
        "candidate": {"version": "V23", "base_checkpoint": "V19/V22", "training": False,
                      "change": "Add 20 mm to the camera-corrected descend and close TCP targets; keep XY, yaw, and delivery actions fixed.",
                      "parameters": 427798},
        "pilot": {"seed_range": [675000, 675011], "v22": summarize_run(base),
                  "offset_tests": pilot_offsets, "selected_offset_m": .02,
                  "selection_note": "The +20 mm candidate was selected from this pilot only. The fresh test uses new seeds."},
    }
    v22_path = TEST / "v22-676000-120.json"
    v23_path = TEST / "v23-plus-20mm-676000-120.json"
    if v22_path.is_file() and v23_path.is_file():
        old, new = load(v22_path), load(v23_path)
        if old.get("complete") and new.get("complete"):
            result["independent_test"] = compare(old, new)
            result["independent_test"]["protocol"] = (
                "120 fresh matched scenes; seeds 676000–676119; fixed V22/V23 policies; "
                "V23 adds 20 mm to camera-corrected descend and close waypoints.")
    destination = ROOT / "portfolio" / "b601-v23-height-evaluation.json"
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(destination),
                      "pilot_placements": {key: value["v23"]["cubes_placed"]
                                           for key, value in pilot_offsets.items()},
                      "independent_test_complete": "independent_test" in result}, indent=2))


if __name__ == "__main__":
    main()

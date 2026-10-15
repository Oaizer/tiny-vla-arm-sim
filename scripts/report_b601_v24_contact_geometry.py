"""Summarise the matched B601 V23/V24 grasp-contact geometry experiment."""
import json
from pathlib import Path

import numpy as np

from report_b601_v22 import exact_sign_p

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts" / "b601-v24-lift-audit"
BASE = ARTIFACTS / "v23-plus-20mm-678000-60.json"
CANDIDATE = ARTIFACTS / "v24-plus-30mm-678000-60.json"
DATA = ROOT / "portfolio" / "b601-v24-contact-geometry.json"
REPORT = ROOT / "docs" / "B601_V24_CONTACT_GEOMETRY.md"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    base, candidate = load(BASE), load(CANDIDATE)
    old, new = base["policy"]["records"], candidate["policy"]["records"]
    assert len(old) == len(new) == 60
    assert all(a["seed"] == b["seed"] for a, b in zip(old, new))

    placements = [0, 0, 0]
    sorts = [0, 0, 0]
    paired = []
    contact_deltas = []
    paired_contact_means = []
    abs_z = {"v23": [], "v24": []}
    contacts = {"v23": [], "v24": []}
    missing_pairs = 0
    for left, right in zip(old, new):
        delta = right["objects_placed"] - left["objects_placed"]
        placements[0] += delta > 0
        placements[1] += delta < 0
        placements[2] += delta == 0
        delta = int(right["success"]) - int(left["success"])
        sorts[0] += delta > 0
        sorts[1] += delta < 0
        sorts[2] += delta == 0
        paired.append({"seed": left["seed"], "v23_placed": left["objects_placed"],
                       "v24_placed": right["objects_placed"],
                       "v23_full_sort": left["success"], "v24_full_sort": right["success"]})
        for a, b in zip(left["objects"], right["objects"]):
            ca, cb = a["grasp_audit"]["contacts"], b["grasp_audit"]["contacts"]
            contacts["v23"].extend(ca)
            contacts["v24"].extend(cb)
            abs_z["v23"].extend(abs(row["cube_local_contact_m"][2]) for row in ca)
            abs_z["v24"].extend(abs(row["cube_local_contact_m"][2]) for row in cb)
            if not ca or not cb:
                missing_pairs += 1
                continue
            mean_a = np.mean([row["corner_proximity"] for row in ca])
            mean_b = np.mean([row["corner_proximity"] for row in cb])
            paired_contact_means.append((mean_a, mean_b))
            contact_deltas.append(mean_b - mean_a)

    d = np.asarray(contact_deltas)
    rng = np.random.default_rng(20261015)
    bootstrap = np.asarray([np.mean(rng.choice(d, len(d), replace=True)) for _ in range(5000)])
    signs = rng.choice([-1, 1], size=(30000, len(d)))
    p_contact = (int(np.sum(np.abs(np.mean(signs * d, axis=1)) >= abs(d.mean()))) + 1) / 30001
    improved, worse = int(np.sum(d < 0)), int(np.sum(d > 0))

    def contact_totals(rows):
        return {
            "finger_contacts": len(rows),
            "mean_corner_proximity_per_contact": round(float(np.mean([r["corner_proximity"] for r in rows])), 4),
            "contacts_within_10mm_of_vertical_face_center": sum(abs(r["cube_local_contact_m"][2]) <= .01 for r in rows),
            "contacts_within_5mm_of_top_or_bottom_rim": sum(abs(r["cube_local_contact_m"][2]) >= .015 for r in rows),
            "mean_absolute_cube_local_contact_z_mm": round(float(np.mean([abs(r["cube_local_contact_m"][2]) for r in rows])) * 1000, 2),
        }

    result = {
        "title": "B601 V24 face-contact geometry test",
        "simulator_only": True,
        "protocol": {
            "seed_range": [old[0]["seed"], old[-1]["seed"]],
            "trials": 60,
            "paired": True,
            "checkpoint": str(candidate["checkpoint"]),
            "change": "Raise camera-corrected descend and close waypoints by 30 mm instead of 20 mm.",
            "training": False,
            "note": "The offset was selected after a small pilot. These 60 matched seeds were not used in that pilot.",
        },
        "task_results": {
            "v23": {"cubes_placed": sum(r["objects_placed"] for r in old), "cubes_attempted": 180,
                    "full_sorts": sum(r["success"] for r in old)},
            "v24": {"cubes_placed": sum(r["objects_placed"] for r in new), "cubes_attempted": 180,
                    "full_sorts": sum(r["success"] for r in new)},
            "paired_cube_placement": {"v24_wins": placements[0], "v24_losses": placements[1],
                                      "ties": placements[2], "exact_sign_test_two_sided_p": exact_sign_p(placements[0], placements[1])},
            "paired_full_sorts": {"v24_wins": sorts[0], "v24_losses": sorts[1],
                                  "ties": sorts[2], "exact_sign_test_two_sided_p": exact_sign_p(sorts[0], sorts[1])},
            "joint_limit_violations": {"v23": sum(not r["joint_limits_respected"] for r in old),
                                       "v24": sum(not r["joint_limits_respected"] for r in new)},
        },
        "contact_geometry": {
            "definition": "Corner proximity is min(|cube-local x|, |cube-local y|) / 21 mm, clipped to 0–1. Lower values place contacts farther from cube corners and nearer the in-plane middle of a side face.",
            "paired_objects_with_contacts_in_both_runs": len(d),
            "paired_objects_missing_contact_in_one_or_both_runs": missing_pairs,
            "v23_mean_object_corner_proximity": round(float(np.mean([x[0] for x in paired_contact_means])), 4),
            "v24_mean_object_corner_proximity": round(float(np.mean([x[1] for x in paired_contact_means])), 4),
            "paired_contact_proximity": {"v24_lower": improved, "v24_higher": worse, "ties": int(np.sum(d == 0)),
                                         "mean_change_v24_minus_v23": round(float(d.mean()), 5),
                                         "bootstrap_95_percent_interval": [round(float(x), 5) for x in np.quantile(bootstrap, [.025, .975])],
                                         "paired_sign_flip_permutation_p": round(p_contact, 7)},
            "v23_contacts": contact_totals(contacts["v23"]),
            "v24_contacts": contact_totals(contacts["v24"]),
        },
        "paired_seed_results": paired,
        "video": {
            "seed": 678001,
            "v23": "media/b601-rs-v23-v24-face-contact-seed-678001-v23.mp4",
            "v24": "media/b601-rs-v23-v24-face-contact-seed-678001-v24.mp4",
        },
    }
    DATA.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    report = [
        "# B601 V24 face-contact geometry test", "",
        "This simulation-only test changes one setting. V24 raises the camera-corrected descend and close waypoints by 30 mm. V23 uses 20 mm. The checkpoint and policy weights stay fixed.", "",
        "## Results", "",
        "The matched test used seeds 678000–678059. V23 placed 129/180 cubes and completed 23/60 full sorts. V24 placed 134/180 cubes and completed 22/60 full sorts. Placement outcomes were 15 V24 wins, 15 losses, and 30 ties (p = 1.00). Full sorts were 12 wins, 13 losses, and 35 ties (p = 1.00). This test does not show a task-success gain.", "",
        f"Among {len(d)} matched objects where both policies made finger contact, V24 had lower corner proximity in {improved}, higher proximity in {worse}, and tied in {int(np.sum(d == 0))}. Mean object-level corner proximity changed from 0.4193 to 0.2949 (mean change {d.mean():.4f}; paired bootstrap 95% interval {np.quantile(bootstrap, .025):.4f} to {np.quantile(bootstrap, .975):.4f}; sign-flip p = {p_contact:.5f}). In this audit, lower values mean contact closer to the in-plane middle of a cube side and farther from its corners.", "",
        "Vertical contact stayed near the top or bottom rim. Contacts within 10 mm of the vertical face center fell from 26/442 to 11/469; mean absolute local contact height changed from 18.71 mm to 19.14 mm. The fixed close waypoint still places the fingertips high on the cube. This suggests that yaw and horizontal centering do not fully set contact height; it does not prove a physical robot cause. Keep V23 as the task-success lead; record V24 as an in-plane grasp-geometry result.", "",
        "All trials used the B601-RS simulation. Contact points are Bullet simulation measurements, not physical tactile-sensor readings. The corner-proximity measure is a geometric proxy and does not establish safer or more reliable contact on real hardware.", "",
        "## Clips", "",
        "The [V23 clip](../portfolio/media/b601-rs-v23-v24-face-contact-seed-678001-v23.mp4) and [V24 clip](../portfolio/media/b601-rs-v23-v24-face-contact-seed-678001-v24.mp4) show the same 3/3 seed.", "",
        "Raw matched results and per-object contact measurements are in [the portfolio data](../portfolio/b601-v24-contact-geometry.json).", "",
    ]
    REPORT.write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({"data": str(DATA), "report": str(REPORT), "contact_gain_p": p_contact,
                      "placement_p": exact_sign_p(placements[0], placements[1]),
                      "full_sort_p": exact_sign_p(sorts[0], sorts[1])}, indent=2))


if __name__ == "__main__":
    main()
